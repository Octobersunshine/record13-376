import numpy as np
from typing import Union, Tuple, Dict, Optional, List


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
    def _build_error(message: str, suggestion: str = "") -> Dict[str, str]:
        """构建统一格式的错误响应字典。

        Args:
            message: 错误描述
            suggestion: 修复建议

        Returns:
            包含 error 和 suggestion 键的字典
        """
        result = {"error": message}
        if suggestion:
            result["suggestion"] = suggestion
        return result

    @staticmethod
    def _validate_inputs(
        matrix_a: np.ndarray, matrix_b: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, str]]]:
        """验证输入矩阵的形状和类型。

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵

        Returns:
            (matrix_a, matrix_b, None) 验证通过
            (None, None, error_dict) 验证失败，返回错误信息
        """
        try:
            matrix_a = np.asarray(matrix_a, dtype=np.float64)
            matrix_b = np.asarray(matrix_b, dtype=np.float64)
        except Exception as exc:
            return None, None, MatrixSimilarityService._build_error(
                f"无法将输入转换为 numpy ndarray: {exc}",
                "请确保输入为可转换为 numpy 数组的类型（如 list 或 ndarray）"
            )

        if matrix_a.ndim != 2 or matrix_b.ndim != 2:
            non_2d = []
            if matrix_a.ndim != 2:
                non_2d.append(f"matrix_a.ndim={matrix_a.ndim}")
            if matrix_b.ndim != 2:
                non_2d.append(f"matrix_b.ndim={matrix_b.ndim}")
            return None, None, MatrixSimilarityService._build_error(
                f"输入必须是二维矩阵, {', '.join(non_2d)}",
                "请检查输入数据的维度，确保为二维矩阵（如使用 np.reshape 调整形状）"
            )

        if matrix_a.shape != matrix_b.shape:
            return None, None, MatrixSimilarityService._build_error(
                f"矩阵形状不一致: matrix_a.shape={matrix_a.shape}, matrix_b.shape={matrix_b.shape}",
                "请检查输入矩阵的维度是否匹配，可使用 matrix_a.reshape 或 np.pad 进行调整"
            )

        return matrix_a, matrix_b, None

    def frobenius_distance(
        self,
        matrix_a: Union[np.ndarray, list],
        matrix_b: Union[np.ndarray, list],
        normalize: bool = False,
    ) -> Union[float, Dict[str, str]]:
        """计算两个矩阵之间的 Frobenius 范数距离。

        Frobenius 范数距离定义为：
            ||A - B||_F = sqrt( sum_ij (A_ij - B_ij)^2 )

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵
            normalize: 是否归一化，归一化后除以元素个数的平方根

        Returns:
            Frobenius 范数距离值；若输入无效则返回错误提示字典
        """
        matrix_a, matrix_b, error = self._validate_inputs(matrix_a, matrix_b)
        if error is not None:
            return error

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
    ) -> Union[float, Dict[str, str]]:
        """计算两个矩阵之间的余弦相似度。

        将矩阵展平为向量后计算余弦相似度：
            cos_theta = (A · B) / (||A||_F * ||B||_F)

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵

        Returns:
            余弦相似度值，范围 [-1, 1]；若输入无效则返回错误提示字典
        """
        matrix_a, matrix_b, error = self._validate_inputs(matrix_a, matrix_b)
        if error is not None:
            return error

        fro_a = np.linalg.norm(matrix_a, ord="fro")
        fro_b = np.linalg.norm(matrix_b, ord="fro")

        if fro_a == 0 or fro_b == 0:
            return self._build_error(
                "矩阵的 Frobenius 范数为 0，无法计算余弦相似度",
                "请确保矩阵不全为零，余弦相似度要求向量具有非零长度"
            )

        dot_product = float(np.sum(matrix_a * matrix_b))
        similarity = dot_product / (fro_a * fro_b)

        similarity = max(-1.0, min(1.0, similarity))
        return float(similarity)

    def correlation_coefficient(
        self,
        matrix_a: Union[np.ndarray, list],
        matrix_b: Union[np.ndarray, list],
    ) -> Union[float, Dict[str, str]]:
        """计算两个矩阵之间的皮尔逊相关系数。

        将矩阵展平为向量后计算皮尔逊相关系数：
            rho = E[(A - mu_A)(B - mu_B)] / (sigma_A * sigma_B)

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵

        Returns:
            相关系数值，范围 [-1, 1]；若输入无效则返回错误提示字典
        """
        matrix_a, matrix_b, error = self._validate_inputs(matrix_a, matrix_b)
        if error is not None:
            return error

        flat_a = matrix_a.flatten()
        flat_b = matrix_b.flatten()

        mean_a = np.mean(flat_a)
        mean_b = np.mean(flat_b)

        centered_a = flat_a - mean_a
        centered_b = flat_b - mean_b

        std_a = np.linalg.norm(centered_a)
        std_b = np.linalg.norm(centered_b)

        if std_a == 0 or std_b == 0:
            return self._build_error(
                "矩阵的标准差为 0（所有元素相同），无法计算相关系数",
                "请确保矩阵元素不全相同，相关系数要求矩阵具有非零标准差"
            )

        covariance = float(np.sum(centered_a * centered_b))
        coefficient = covariance / (std_a * std_b)

        coefficient = max(-1.0, min(1.0, coefficient))
        return float(coefficient)

    def compute_all(
        self,
        matrix_a: Union[np.ndarray, list],
        matrix_b: Union[np.ndarray, list],
    ) -> Union[Dict[str, float], Dict[str, str]]:
        """一次性计算所有相似度/距离指标。

        Args:
            matrix_a: 第一个矩阵
            matrix_b: 第二个矩阵

        Returns:
            包含所有指标结果的字典；若输入无效则返回错误提示字典
        """
        _, _, error = self._validate_inputs(matrix_a, matrix_b)
        if error is not None:
            return error

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

    def batch_compare(
        self,
        source: Union[np.ndarray, list],
        candidates: List[Union[np.ndarray, list]],
        labels: Optional[List[str]] = None,
        rank_by: str = "cosine_similarity",
        ascending: Optional[bool] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """将一个源矩阵与多个候选矩阵批量计算相似度，返回排名列表。

        Args:
            source: 源矩阵，将与每个候选矩阵分别比对
            candidates: 候选矩阵列表
            labels: 候选矩阵的标签名称列表，若为 None 则自动生成 "candidate_0", ...
            rank_by: 排名依据的指标，可选值：
                "frobenius_distance", "frobenius_distance_normalized",
                "cosine_similarity", "correlation_coefficient"
            ascending: 是否升序排列。若为 None 则按指标自动选择：
                距离类指标默认升序（越小越相似），相似度类指标默认降序（越大越相似）
            top_k: 仅返回排名前 k 个结果，若为 None 则返回全部

        Returns:
            排名列表，每个元素为字典，包含：
                - rank: 排名（从 1 开始）
                - label: 候选矩阵标签
                - 各指标的值（或错误信息）
                - rank_by_value: 排名依据指标的值
        """
        valid_metrics = {
            "frobenius_distance",
            "frobenius_distance_normalized",
            "cosine_similarity",
            "correlation_coefficient",
        }

        if rank_by not in valid_metrics:
            return [self._build_error(
                f"无效的 rank_by 指标: '{rank_by}'",
                f"可选值为: {', '.join(sorted(valid_metrics))}"
            )]

        if labels is not None and len(labels) != len(candidates):
            return [self._build_error(
                f"labels 长度({len(labels)})与 candidates 长度({len(candidates)})不一致",
                "请确保 labels 与 candidates 数量相同，或传入 labels=None 自动生成"
            )]

        if ascending is None:
            ascending = rank_by in {"frobenius_distance", "frobenius_distance_normalized"}

        if labels is None:
            labels = [f"candidate_{i}" for i in range(len(candidates))]

        results: List[Dict] = []

        for idx, candidate in enumerate(candidates):
            label = labels[idx]
            metrics = self.compute_all(source, candidate)

            entry: Dict = {"label": label}
            entry.update(metrics)

            if "error" in metrics:
                entry["rank_by_value"] = None
            else:
                entry["rank_by_value"] = metrics.get(rank_by)

            results.append(entry)

        valid_results = [r for r in results if r["rank_by_value"] is not None]
        invalid_results = [r for r in results if r["rank_by_value"] is None]

        valid_results.sort(
            key=lambda r: r["rank_by_value"],
            reverse=not ascending,
        )

        sorted_results = valid_results + invalid_results

        for rank_idx, entry in enumerate(sorted_results, start=1):
            entry["rank"] = rank_idx

        if top_k is not None:
            sorted_results = sorted_results[:top_k]

        return sorted_results


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

    mat_mismatch_a = np.random.rand(3, 4)
    mat_mismatch_b = np.random.rand(5, 6)

    mat_1d_a = np.array([1.0, 2.0, 3.0])
    mat_1d_b = np.array([4.0, 5.0, 6.0])

    test_cases = [
        ("两个相同的单位矩阵", mat_ident_a, mat_ident_b),
        ("两个不同的随机矩阵", mat_random_a, mat_random_b),
        ("两个非常相近的矩阵", mat_similar_a, mat_similar_b),
        ("矩阵与其倍数（成比例）", mat_random_a, mat_scaled),
        ("矩阵与其负数矩阵", mat_random_a, mat_negative),
        ("维度不匹配的矩阵", mat_mismatch_a, mat_mismatch_b),
        ("一维数组（非矩阵）", mat_1d_a, mat_1d_b),
    ]

    print("=" * 60)
    print("  单对矩阵相似度计算")
    print("=" * 60)

    for name, a, b in test_cases:
        print(f"\n{'─' * 50}")
        print(f"测试用例: {name}")

        a_display = a.shape if hasattr(a, 'shape') else f"list(len={len(a)})"
        b_display = b.shape if hasattr(b, 'shape') else f"list(len={len(b)})"
        print(f"  matrix_a shape: {a_display}")
        print(f"  matrix_b shape: {b_display}")

        results = service.compute_all(a, b)
        for key, value in results.items():
            print(f"  {key}: {value}")

    np.random.seed(123)
    source = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    candidates = [
        source + 0.01,
        source * 2.0,
        np.random.rand(2, 3),
        -source,
        np.random.rand(2, 3),
        np.random.rand(3, 4),
    ]

    cand_labels = [
        "微小扰动",
        "倍数缩放(×2)",
        "随机矩阵A",
        "负数矩阵",
        "随机矩阵B",
        "维度不匹配",
    ]

    print(f"\n\n{'=' * 60}")
    print("  批量比对：按余弦相似度排名（默认）")
    print(f"{'=' * 60}")
    print(f"  源矩阵 shape: {source.shape}\n")

    ranking = service.batch_compare(
        source, candidates, labels=cand_labels,
        rank_by="cosine_similarity",
    )

    for entry in ranking:
        if "error" in entry and "cosine_similarity" not in entry:
            print(f"  #{entry['rank']} {entry['label']}: {entry.get('error', 'N/A')}")
        else:
            print(
                f"  #{entry['rank']} {entry['label']}: "
                f"cosine={entry['cosine_similarity']:.4f}, "
                f"corr={entry['correlation_coefficient']:.4f}, "
                f"fro_dist={entry['frobenius_distance']:.4f}"
            )

    print(f"\n\n{'=' * 60}")
    print("  批量比对：按 Frobenius 距离排名（仅 Top 3）")
    print(f"{'=' * 60}")

    ranking_top3 = service.batch_compare(
        source, candidates, labels=cand_labels,
        rank_by="frobenius_distance",
        top_k=3,
    )

    for entry in ranking_top3:
        print(
            f"  #{entry['rank']} {entry['label']}: "
            f"fro_dist={entry['frobenius_distance']:.4f}, "
            f"fro_dist_norm={entry['frobenius_distance_normalized']:.4f}"
        )


if __name__ == "__main__":
    main()
