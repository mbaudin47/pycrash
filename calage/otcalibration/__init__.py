"""otcalibration module."""
from .daCalibrationFunction import DaCalibrationFunction
from .daVectorizedCalibrationFunction import DaVectorizedCalibrationFunction
from .da3DVAR import Da3DVAR
from .daBLUE import DaBLUE
from .daLinear import DaLinear
from .daNLLS import DaNLLS
from .dalib import plotModelVsDataBeforeAndAFter
from .dalib import plotObservationsVsPredictionsBeforeAfter
from .dalib import plotResiduals
from .HellingerDistanceAlgorithm import HellingerDistanceAlgorithm
from .LinearLeastSquaresCalibrationRandomVector import LinearLeastSquaresCalibrationRandomVector
from .NonLinearLeastSquaresCalibrationRandomVector import NonLinearLeastSquaresCalibrationRandomVector
from .LinearGaussianCalibrationRandomVector import LinearGaussianCalibrationRandomVector
from .NonLinearGaussianCalibrationRandomVector import NonLinearGaussianCalibrationRandomVector
from .caliblib import (
    gaussianLinearCalibrationFromCholesky,
    gaussianCalibrationCostFunction,
    squaredMahalanobis,
    gaussianLinearCalibrationFromKalman,
    gaussianLinearCalibrationFromCholeskyFixed,
    computeJacobianParameterMatrix,
    computeGlobalObservationsCovarianceMatrixFromLocalCovariance,
    computeKalmanMatrix,
)
from .LinearGaussianMonteCarloCalibration import LinearGaussianMonteCarloCalibration
from .LinearLeastSquaresValidation import LinearLeastSquaresValidation
from .UnivariatePolynomialParametricModelFactory import UnivariatePolynomialParametricModelFactory

__version__ = "1.0"

__all__ = [
    "gaussianLinearCalibrationFromCholesky",
    "gaussianCalibrationCostFunction",
    "squaredMahalanobis",
    "gaussianLinearCalibrationFromKalman",
    "gaussianLinearCalibrationFromCholeskyFixed",
    "computeJacobianParameterMatrix",
    "computeGlobalObservationsCovarianceMatrixFromLocalCovariance",
    "DaCalibrationFunction",
    "DaVectorizedCalibrationFunction",
    "Da3DVAR",
    "DaBLUE",
    "DaLinear",
    "DaNLLS",
    "plotModelVsDataBeforeAndAFter",
    "plotObservationsVsPredictionsBeforeAfter",
    "plotResiduals",
    "HellingerDistanceAlgorithm",
    "LinearLeastSquaresCalibrationRandomVector",
    "NonLinearLeastSquaresCalibrationRandomVector",
    "LinearGaussianCalibrationRandomVector",
    "NonLinearGaussianCalibrationRandomVector",
    "LinearGaussianMonteCarloCalibration",
    "UnivariatePolynomialParametricModelFactory", 
    "LinearLeastSquaresValidation"
]
