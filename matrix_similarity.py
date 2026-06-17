import numpy as np
from typing import Union, Tuple, Dict


class MatrixSimilarityService:
    """矩阵相似度计算服务。

    支持计算两个矩阵之间的：
        - Frobenius 范数距离
        - 余弦相似度
        - 矩阵相关系数
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _validate_inputs(
        matrix_a: np.ndarray, matrix_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """验证输入矩阵的形状和类型。

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵

        Returns:
            转换为 np.ndarray 后的两个矩阵

        Raises:
            ValueError: 当两个矩阵形状不一致时
            TypeError: 当输入无法转换为 ndarray 时
        """
        try:
            matrix_a = np.asarray(matrix_a, dtype=np.float64)
            matrix_b = np.asarray(matrix_b, dtype=np.float64)
        except Exception as exc:
            raise TypeError(
                f"无法将输入转换为 numpy ndarray: {exc}"
            ) from exc

        if matrix_a.shape != matrix_b.shape:
            raise ValueError(
                f"矩阵形状不一致: matrix_a.shape={matrix_a.shape}, "
                f"matrix_b.shape={matrix_b.shape}"
            )

        if matrix_a.ndim != 2:
            raise ValueError(
                f"输入必须是二维矩阵, 当前维度: matrix_a.ndim={matrix_a.ndim}"
            )

        return matrix_a, matrix_b

    def frobenius_distance(
        self,
        matrix_a: Union[np.ndarray, list],
        matrix_b: Union[np.ndarray, list],
        normalize: bool = False,
    ) -> float:
        """计算两个矩阵之间的 Frobenius 范数距离。

        Frobenius 范数距离定义为：
            ||A - B||_F = sqrt( sum_ij (A_ij - B_ij)^2 )

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵
            normalize: 是否归一化，归一化后除以元素个数的平方根

        Returns:
            Frobenius 范数距离值
        """
        matrix_a, matrix_b = self._validate_inputs(matrix_a, matrix_b)
        diff = matrix_a - matrix_b
        distance = float(np.linalg.norm(diff, ord="fro"))

        if normalize:
            n_elements = matrix_a.size
            distance = distance / np.sqrt(n_elements)

        return distance

    def cosine_similarity(
        self,
        matrix_a: Union[np.ndarray, list],
        matrix_b: Union[np.ndarray, list],
    ) -> float:
        """计算两个矩阵之间的余弦相似度。

        将矩阵展平为向量后计算余弦相似度：
            cos_theta = (A · B) / (||A||_F * ||B||_F)

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵

        Returns:
            余弦相似度值，范围 [-1, 1]
        """
        matrix_a, matrix_b = self._validate_inputs(matrix_a, matrix_b)

        fro_a = np.linalg.norm(matrix_a, ord="fro")
        fro_b = np.linalg.norm(matrix_b, ord="fro")

        if fro_a == 0 or fro_b == 0:
            raise ZeroDivisionError(
                "矩阵的 Frobenius 范数为 0，无法计算余弦相似度"
            )

        dot_product = float(np.sum(matrix_a * matrix_b))
        similarity = dot_product / (fro_a * fro_b)

        similarity = max(-1.0, min(1.0, similarity))
        return float(similarity)

    def correlation_coefficient(
        self,
        matrix_a: Union[np.ndarray, list],
        matrix_b: Union[np.ndarray, list],
    ) -> float:
        """计算两个矩阵之间的皮尔逊相关系数。

        将矩阵展平为向量后计算皮尔逊相关系数：
            rho = E[(A - mu_A)(B - mu_B)] / (sigma_A * sigma_B)

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵

        Returns:
            相关系数值，范围 [-1, 1]
        """
        matrix_a, matrix_b = self._validate_inputs(matrix_a, matrix_b)

        flat_a = matrix_a.flatten()
        flat_b = matrix_b.flatten()

        mean_a = np.mean(flat_a)
        mean_b = np.mean(flat_b)

        centered_a = flat_a - mean_a
        centered_b = flat_b - mean_b

        std_a = np.linalg.norm(centered_a)
        std_b = np.linalg.norm(centered_b)

        if std_a == 0 or std_b == 0:
            raise ZeroDivisionError(
                "矩阵的标准差为 0（所有元素相同），无法计算相关系数"
            )

        covariance = float(np.sum(centered_a * centered_b))
        coefficient = covariance / (std_a * std_b)

        coefficient = max(-1.0, min(1.0, coefficient))
        return float(coefficient)

    def compute_all(
        self,
        matrix_a: Union[np.ndarray, list],
        matrix_b: Union[np.ndarray, list],
    ) -> Dict[str, float]:
        """一次性计算所有相似度/距离指标。

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵

        Returns:
            包含所有指标结果的字典
        """
        return {
            "frobenius_distance": self.frobenius_distance(matrix_a, matrix_b),
            "frobenius_distance_normalized": self.frobenius_distance(
                matrix_a, matrix_b, normalize=True
            ),
            "cosine_similarity": self.cosine_similarity(matrix_a, matrix_b),
            "correlation_coefficient": self.correlation_coefficient(
                matrix_a, matrix_b
            ),
        }


def main() -> None:
    """示例：演示矩阵相似度服务的使用方法。"""
    service = MatrixSimilarityService()

    np.random.seed(42)

    mat_ident_a = np.eye(3)
    mat_ident_b = np.eye(3)

    mat_random_a = np.random.rand(4, 5)
    mat_random_b = np.random.rand(4, 5)

    mat_similar_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    mat_similar_b = mat_similar_a + 0.01

    mat_scaled = mat_random_a * 2.5

    mat_negative = -mat_random_a

    test_cases = [
        ("两个相同的单位矩阵", mat_ident_a, mat_ident_b),
        ("两个不同的随机矩阵", mat_random_a, mat_random_b),
        ("两个非常相近的矩阵", mat_similar_a, mat_similar_b),
        ("矩阵与其倍数（成比例）", mat_random_a, mat_scaled),
        ("矩阵与其负数矩阵", mat_random_a, mat_negative),
    ]

    for name, a, b in test_cases:
        print(f"\n{'=' * 50}")
        print(f"测试用例: {name}")
        print(f"  matrix_a shape: {a.shape}")
        print(f"  matrix_b shape: {b.shape}")
        print(f"{'=' * 50}")

        results = service.compute_all(a, b)
        for key, value in results.items():
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
