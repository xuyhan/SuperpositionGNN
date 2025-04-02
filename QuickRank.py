import numpy as np


def rank_analysis(matrix, tol=1e-5):
    """
    Performs rank analysis on the given matrix using Singular Value Decomposition (SVD).

    Args:
        matrix (np.ndarray): The input matrix to analyze.
        tol (float): Threshold below which singular values are considered zero.

    Returns:
        dict: Contains the singular values, numerical rank, and effective dimension.
    """
    U, singular_values, VT = np.linalg.svd(matrix)

    numerical_rank = np.sum(singular_values > tol)

    print(f"Singular Values: {singular_values}")
    print(f"Numerical Rank (tol={tol}): {numerical_rank}")
    print(f"Eigenvectors (V): {VT}")

    return {
        'singular_values': singular_values,
        'numerical_rank': numerical_rank,
        'effective_dimension': numerical_rank
    }


# Example usage:
if __name__ == "__main__":
    # Example arbitrary matrix
    example_matrix = np.array([
             [
                    3.720973014831543,
                    0.6151129603385925,
                    0.3927508294582367,
                    -1.2994953393936157,
                    -0.06813999265432358,
                    -2.282247543334961
                ],
                [
                    0.44725367426872253,
                    -1.056020736694336,
                    0.00868317298591137,
                    -1.1549776792526245,
                    0.3620094656944275,
                    -0.8370203375816345
                ],
                [
                    1.0047297477722168,
                    -1.8882521390914917,
                    -0.23850736021995544,
                    2.5852596759796143,
                    -0.45086896419525146,
                    0.27138325572013855
                ],
                [
                    -0.36161959171295166,
                    1.5696903467178345,
                    -0.3614432215690613,
                    0.6902425289154053,
                    -0.30354416370391846,
                    -1.964904546737671
                ],
                [
                    1.444005012512207,
                    -0.7049777507781982,
                    -0.07589370757341385,
                    -0.9503803253173828,
                    0.3560221195220947,
                    -0.5438014268875122
                ],
                [
                    -0.7056887745857239,
                    -0.7721424698829651,
                    -0.3352954387664795,
                    1.1670470237731934,
                    -0.14908716082572937,
                    -1.2029991149902344
                ]])

    analysis_result = rank_analysis(example_matrix)
    print(analysis_result)