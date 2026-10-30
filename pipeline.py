"""
tft_analyzer_pipeline.py
================================================================================
프로젝트명: Sol-Gel 산화물 TFT 파라미터 자동 추출 및 불량 진단 파이프라인
작성 목적: 
  - 엑셀 기반 수작업 분석의 한계(이상치 오염, 휴먼 에러, 수식 붕괴) 해결
  - Sol-Gel 용액 공정 특유의 극심한 산포 및 비이상적 결함 데이터 방어
  - 수치 기반 자동 진단 및 AI(LLM) 추론 연동을 위한 표준 데이터 구조화
================================================================================
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression


# ==============================================================================
# 1. 데이터 모델 정의 (Data Structure)
# ==============================================================================
@dataclass
class DeviceMetrics:
    """
    단일 소자(TFT)에서 추출된 물리적/전기적 특성 지표를 담는 표준 데이터 규격.
    이 객체는 후속 통계 분석 및 LLM API(AI 진단) 호출 시의 Payload(입력 데이터)로 직렬화됩니다.
    """
    sample_id: str          # 샘플 고유 식별자 (예: IGZO_Al2O3_W500_L50_Sample1)
    condition: str          # 공정 조건 그룹 (예: IGZO_Al2O3, InOx_SiO2)
    channel_l_um: float     # 채널 길이 (Length, 단위: um)
    channel_w_um: float     # 채널 폭 (Width, 단위: um)
    vth: float              # 문턱 전압 (Threshold Voltage, 단위: V)
    ss_mv_dec: float        # 서브스레숄드 스윙 (Subthreshold Swing, 단위: mV/dec)
    mobility: float         # 포화 전계효과 이동도 (Saturation Mobility, 단위: cm^2/V·s)
    ion_ioff_ratio: float   # On/Off 전류 점멸비 (I_on / I_off)
    r2_score: float         # Vth 외삽 구간 선형 회귀 결정계수 (피팅 신뢰도, 1에 가까울수록 이상적)
    valid: bool             # 데이터 유효성 여부 (True: 정상 소자, False: 결함/불량 소자)
    status_msg: str         # 소자 상태 메시지 또는 불량 유형 태그 (예: NORMAL, INVALID_OPEN)


# ==============================================================================
# 2. 데이터 품질 검증 게이트 (Data Quality Gate)
# ==============================================================================
class DataQualityGate:
    """
    [물리적 스위칭 사전 검증 게이트]
    - 목적: 미분(SS)이나 선형 회귀(Vth) 알고리즘을 무작정 돌리기 전, 
            물리적으로 트랜지스터 동작이 성립하는지 먼저 검사하여 
            NaN 에러나 비정상 수치 폭주를 원천 차단합니다.
    - 배경: Sol-Gel 공정에서는 스핀 코팅 중 미세 먼지, 습도 변화, 불완전 산화로 인해
            단선(Open)이나 절연막 쇼트(Breakdown)가 빈번히 발생합니다.
    """

    @staticmethod
    def validate(vgs: np.ndarray, id_curr: np.ndarray) -> Tuple[bool, str]:
        """
        I-V 원시 곡선을 전수 조사하여 소자의 측정 무결성을 판정합니다.
        
        매개변수:
            vgs: 게이트 전압 배열 [V]
            id_curr: 측정된 드레인 전류 배열 [A]
        반환값:
            (유효 여부: bool, 판정 메시지: str)
        """
        # [검증 1] 측정 데이터 포인트 개수 확인
        # 전압 스위프 구간의 데이터가 너무 적으면 미분/회귀 연산 자체가 불가능합니다.
        if len(vgs) < 10 or len(id_curr) < 10:
            return False, "INVALID_DATA_LENGTH: 측정 데이터 포인트가 너무 적음 (< 10 pt)"

        max_id = np.max(id_curr)
        min_id = np.min(id_curr)

        # [검증 2] 단선/미도통 결함 (Open Failure) 판정
        # 게이트 전압을 끝까지 걸었는데도 최대 도통 전류가 계측기 바닥(100 pA 미만)이라면
        # 전극 프로빙이 빗나갔거나, 채널 활성층이 아예 연결되지 않은 시편입니다.
        if max_id < 1e-10:
            return False, "INVALID_OPEN: 최대 On-전류 < 100 pA (채널 미형성 또는 프로빙 접촉 불량)"

        # [검증 3] 절연막 파괴 / 저항성 쇼트 (Breakdown Failure) 판정
        # 트랜지스터는 게이트로 전류를 켜고 끌 수 있어야(Switching) 합니다.
        # On/Off 비율이 10배도 되지 않는다는 것은 게이트 절연막이 뚫렸거나(Pinhole)
        # 활성층이 반도체가 아닌 단순 도체 저항처럼 작동하고 있음을 의미합니다.
        ion_ioff = max_id / (min_id + 1e-13)
        if ion_ioff < 10.0:
            return False, "INVALID_BREAKDOWN: On/Off 비 < 10 (스위칭 상실, 절연막 파괴 또는 저항성 단락)"

        # 모든 방어 기준을 통과하면 정상 연산 대상으로 승인
        return True, "VALID"


# ==============================================================================
# 3. 파라미터 정밀 추출 엔진 (Parameter Extractor)
# ==============================================================================
class TFTParameterExtractor:
    """
    [소자 물리 파라미터 정밀 추출 엔진]
    - 입력받은 I-V 곡선에서 Subthreshold Swing(SS), 문턱전압(Vth), 전계효과 이동도(mu)를 계산합니다.
    - 계측기 미세 노이즈를 걸러내기 위해 Savitzky-Golay 필터를 적용하며,
      포화 영역 공식(Saturation current equation)에 따라 20%~80% 구간 선형 외삽을 수행합니다.
    """

    def __init__(self, ci_f_cm2: float = 1.15e-8, w_um: float = 500.0):
        """
        초기화 파라미터:
            ci_f_cm2: 게이트 절연막 단위 면적당 커패시턴스 [F/cm^2]
                      (예: High-k Al2O3 ~ 1.15e-8, Low-k SiO2 ~ 3.45e-9)
            w_um: 채널 폭 (Width) [um]
        """
        self.ci = ci_f_cm2
        self.w_um = w_um

    def extract(self, vgs: np.ndarray, id_curr: np.ndarray, sample_id: str, condition: str, l_um: float) -> DeviceMetrics:
        """
        단일 Transfer Curve (Vgs-Id)로부터 물리 지표를 추출합니다.
        """
        # 1. 수치 전처리:
        # 로그 스케일 변환 시 음수 전류(노이즈로 튄 값)가 들어가 log(0) 에러가 나는 것을 방지하기 위해
        # 계측기 노이즈 플로어인 1e-13 A (0.1 pA)를 하한선으로 클램핑합니다.
        id_clean = np.maximum(id_curr, 1e-13)
        log_id = np.log10(id_clean)
        sqrt_id = np.sqrt(id_clean)
        ion_ioff = np.max(id_clean) / np.min(id_clean)

        # 2. 품질 게이트 검증: 비정상 소자 여부를 사전에 걸러냅니다.
        is_valid, status_msg = DataQualityGate.validate(vgs, id_clean)
        if not is_valid:
            return DeviceMetrics(
                sample_id=sample_id,
                condition=condition,
                channel_l_um=l_um,
                channel_w_um=self.w_um,
                vth=np.nan,
                ss_mv_dec=np.nan,
                mobility=np.nan,
                ion_ioff_ratio=round(float(ion_ioff), 2),
                r2_score=0.0,
                valid=False,
                status_msg=status_msg
            )

        # 3. Subthreshold Swing (SS) 추출:
        # SS = dVgs / d(log10 Id) [mV/dec]
        # 실측 데이터는 바닥 전류 구간에서 미세 노이즈가 심하므로, 
        # 단순 차분 대신 Savitzky-Golay 다항식 필터로 스무딩한 후 기울기(미분값)를 구합니다.
        try:
            if len(log_id) >= 9:
                log_smooth = savgol_filter(log_id, window_length=9, polyorder=2)
                dlog = np.gradient(log_smooth, vgs)
            else:
                dlog = np.gradient(log_id, vgs)

            # 노이즈 바닥(10 pA)과 강한 턴온 전(0.1 uA) 사이의 "실제 서브스레숄드 전도 구간"만 마스킹
            valid_sub_mask = (id_clean >= 1e-11) & (id_clean <= 1e-7)
            if np.any(valid_sub_mask) and np.max(dlog[valid_sub_mask]) > 0:
                max_slope = np.max(dlog[valid_sub_mask]) # 최대 기울기 (가장 가파른 턴온 지점)
                ss_val = (1.0 / max_slope) * 1000.0      # V/dec -> mV/dec 단위 변환
            else:
                ss_val = np.nan
        except Exception:
            ss_val = np.nan

        # 4. 문턱 전압(Vth) 및 이동도(Mobility) 추출:
        # 포화 영역 수식: sqrt(Id) = sqrt( (W * Ci * mu) / (2 * L) ) * (Vgs - Vth)
        # 따라서 sqrt(Id) 대 Vgs 그래프에서 선형 피팅선의 x절편이 Vth가 되고, 
        # 기울기의 제곱으로부터 전계효과 이동도(mu)를 산출합니다.
        try:
            # 바닥의 험프 구간(0~20%)과 고전압 접촉저항 굽음 구간(80~100%)을 피해
            # 20% ~ 80% Quantile의 안정적인 선형 구간만 추출하여 회귀 분석을 진행합니다.
            p20, p80 = np.quantile(sqrt_id, 0.2), np.quantile(sqrt_id, 0.8)
            fit_mask = (sqrt_id >= p20) & (sqrt_id <= p80)
            X = vgs[fit_mask].reshape(-1, 1)
            y = sqrt_id[fit_mask]

            if len(X) >= 3:
                reg = LinearRegression().fit(X, y)
                slope = float(reg.coef_[0])
                intercept = float(reg.intercept_)
                r2 = float(reg.score(X, y))

                # x절편 계산: y = slope * x + intercept = 0  =>  x = -intercept / slope
                vth_val = -intercept / slope if slope > 0 else np.nan
                
                # 이동도 계산: mu = (slope^2 * 2 * L) / (W * Ci)
                w_cm = self.w_um * 1e-4
                l_cm = l_um * 1e-4
                mobility_val = ((slope ** 2) * 2.0 * l_cm) / (w_cm * self.ci) if slope > 0 else np.nan
            else:
                vth_val, mobility_val, r2 = np.nan, np.nan, 0.0
        except Exception:
            vth_val, mobility_val, r2 = np.nan, np.nan, 0.0

        # 5. 수식 오류로 인한 비현실적 이상치(Spike) 필터링:
        # (실제 과거 엑셀 데이터에서 InOx2 샘플의 이동도가 수천 cm^2/Vs로 튀었던 현상 방어)
        # 산화물 반도체의 물리적 한계치(일반적으로 100~200 cm^2/Vs 이하)를 초과하면 이상치로 판정합니다.
        if np.isnan(mobility_val) or mobility_val > 200.0:
            return DeviceMetrics(
                sample_id=sample_id,
                condition=condition,
                channel_l_um=l_um,
                channel_w_um=self.w_um,
                vth=round(float(vth_val), 2) if not np.isnan(vth_val) else np.nan,
                ss_mv_dec=round(float(ss_val), 1) if not np.isnan(ss_val) else np.nan,
                mobility=round(float(mobility_val), 2) if not np.isnan(mobility_val) else np.nan,
                ion_ioff_ratio=round(float(ion_ioff), 2),
                r2_score=round(r2, 4),
                valid=False,
                status_msg="INVALID_MOBILITY_SPIKE: 수식 오차로 인한 비정상 이동도 폭주"
            )

        # 정상 판정된 소자 객체 반환
        return DeviceMetrics(
            sample_id=sample_id,
            condition=condition,
            channel_l_um=l_um,
            channel_w_um=self.w_um,
            vth=round(float(vth_val), 2),
            ss_mv_dec=round(float(ss_val), 1),
            mobility=round(float(mobility_val), 2),
            ion_ioff_ratio=round(float(ion_ioff), 2),
            r2_score=round(r2, 4),
            valid=True,
            status_msg="NORMAL"
        )


# ==============================================================================
# 4. 강건한 통계 집계 엔진 (Robust Aggregator)
# ==============================================================================
class RobustAggregator:
    """
    [이상치 저항 통계 엔진]
    - 배경: 엑셀 단순 AVERAGE 수식은 3개 샘플 중 1개만 튀어도 전체 평균을 완전히 오염시킵니다.
    - 해결: 불량 시편(valid == False)을 통계에서 사전 배제하고,
            극단치 왜곡에 강건한 중앙값(Median)과 정규화 지표를 그룹별로 집계합니다.
    """

    @staticmethod
    def aggregate(metrics_list: List[DeviceMetrics]) -> pd.DataFrame:
        """
        전체 추출 결과 리스트를 받아 공정 조건 및 채널 규격별로 정리된 요약 통계표를 반환합니다.
        """
        df = pd.DataFrame([asdict(m) for m in metrics_list])
        
        # 유효 판정을 받은 샘플만 선별 (Quality Gate 통과 샘플)
        valid_df = df[df["valid"] == True]

        if valid_df.empty:
            print("[Warning] 유효한 정상 소자 데이터가 존재하지 않습니다.")
            return pd.DataFrame()

        # 조건(공정군) 및 채널 길이(L)별 그룹화 집계
        summary = valid_df.groupby(["condition", "channel_l_um"]).agg(
            vth_median=("vth", "median"),              # Vth 중앙값 (V)
            ss_median=("ss_mv_dec", "median"),         # SS 중앙값 (mV/dec)
            mobility_median=("mobility", "median"),     # 이동도 중앙값 (cm^2/Vs)
            r2_mean=("r2_score", "mean"),              # 선형 피팅 평균 결정계수
            valid_count=("sample_id", "count")         # 그룹 내 정상 통과 샘플 수
        ).reset_index()

        return summary
