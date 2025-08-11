import numpy as np


# --- Geometric Analysis Functions ---
def geometry_of_representation(avg_embeddings_active):
    '''
    Returns the geometry of the representation (i.e. whether or not the active embeddings
    are in ideal configuraion).
    The code computes an ideal angle based on n vectors and m dimensions: n≤m returns 90° (orthogonality);
    n=m+1 yields a regular simplex; n>m+1 employs a heuristic, effectively reducing the angle.
    '''
    embeddings = list(avg_embeddings_active.values())
    n = len(embeddings)
    if n == 0:
        return 0, 0
    m = len(embeddings[0])
    if n <= m:
        ideal_angle = np.pi / 2
    elif n == m + 1:
        ideal_angle = np.arccos(-1.0 / m)
    else:
        ideal_angle = np.arccos(np.sqrt((n - m) / (m * (n - 1))))
    tol = 0.6  # Tolerance in radians.
    smallest_angles = []
    collapsed = 0
    for i in range(n):
        angles = []
        v = np.array(embeddings[i])
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            w = np.array(embeddings[j])
            norm_w = np.linalg.norm(w)
            if norm_w == 0:
                continue
            cosine = np.dot(v, w) / (norm_v * norm_w)
            cosine = np.clip(cosine, -1.0, 1.0)
            angle = np.arccos(cosine)
            angles.append(angle)
        if angles:
            min_angle = min(angles)
            smallest_angles.append(min_angle)
            if min_angle < 0.1:
                collapsed += 1
    if all(angle >= (ideal_angle - tol) for angle in smallest_angles):
        geometry = n
    else:
        geometry = 0
    return geometry, collapsed


def active_targets_in_representation(target_dim, avg_predictions, avg_embeddings):
    '''
    Returns the number of active and accurate targets in the representation plus the
    embeddings of the active targets to be used in the geometry analysis.
    '''
    num_active_targets = 0
    num_accurate_targets = 0
    avg_embeddings_active = {}
    sigma_accurate = 0.3
    sigma_active = 0.5

    is_active = []

    for key, preds in avg_predictions.items():
        if all(abs(key[i] - preds[i].item()) < sigma_active for i in range(target_dim)):
            num_active_targets += 1
            avg_embeddings_active[key] = avg_embeddings[key]
            is_active.append(1)
        else:
            is_active.append(0)

    for key, preds in avg_predictions.items():
        if all(abs(key[i] - preds[i].item()) < sigma_accurate for i in range(target_dim)):
            num_accurate_targets += 1
    return num_active_targets, avg_embeddings_active, num_accurate_targets, is_active


def structure_of_representation(target_dim, avg_predictions, avg_embeddings, final_loss):
    '''
    Returns the structure of the representation (i.e. the number of active targets, the number of accurate targets,
    the geometry of the representation, and whether the representation is collapsed).
    '''
    num_active_targets, avg_embeddings_active, num_accurate_targets, is_active = active_targets_in_representation(
        target_dim, avg_predictions, avg_embeddings)
    geometry, collapsed = geometry_of_representation(avg_embeddings_active)
    if geometry > 0:
        category_with_loss = [target_dim, num_active_targets, num_accurate_targets, geometry, collapsed, final_loss,
                              is_active]
        return tuple(category_with_loss)
    else:
        category_with_loss = [target_dim, num_active_targets, num_accurate_targets, "Failed", collapsed, final_loss,
                              is_active]
        return target_dim, num_active_targets, num_accurate_targets, "Failed", collapsed, final_loss, is_active


def geometry_analysis(results, all_model_params, all_average_embeddings):
    config_losses = {}
    model_params = {}
    average_embeddings = {}
    for i, res in enumerate(results):
        if res is None:
            continue
        key = (res[1], res[2], res[3], res[4])
        config_losses.setdefault(key, []).append(res[5])
        model_params.setdefault(key, []).append(all_model_params[i])
        average_embeddings.setdefault(key, []).append(all_average_embeddings[i])
    return config_losses, model_params, average_embeddings


def svd_analysis(embeddings):

    if isinstance(embeddings, dict):
        embeddings = list(embeddings.values())

        embeddings = np.array(
            [embedding.cpu().detach().numpy() if hasattr(embedding, 'cpu') else embedding.numpy() for embedding in
             embeddings])

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    rank = np.linalg.matrix_rank(embeddings)

    U, singular_values, Vt = np.linalg.svd(embeddings, full_matrices=False)

    return rank, singular_values
