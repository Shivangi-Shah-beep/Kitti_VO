# utils.py
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def plot_trajectories(gt_traj=None, traj2D=None, traj3D=None, save_path=None):
    plt.figure(figsize=(10, 7))

    if traj3D is not None:
        length= len(traj3D)
    
    elif traj2D is not None:
        length= len(traj2D)

    if gt_traj is not None:
        gt_traj = np.array(gt_traj)[:length]
        plt.plot(gt_traj[:, 0], gt_traj[:, 2], label="Ground Truth", linewidth=2)

    if traj2D is not None:
        traj2D = np.array(traj2D)
        plt.plot(traj2D[:, 0], traj2D[:, 2], label="VO 2D (Essential Matrix)", linestyle='--')

    if traj3D is not None:
        traj3D = np.array(traj3D)
        plt.plot(traj3D[:, 0], traj3D[:, 2], label="VO 3D (PnP)", linestyle='-.')

    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def umeyamaAlignment(source_points, target_points, with_scaling=True):
    assert source_points.shape== target_points.shape
    src = o3d.utility.Vector3dVector(source_points)
    tgt = o3d.utility.Vector3dVector(target_points)

    # Estimate transformation
    transformation = o3d.pipelines.registration.registration_umeyama(
        src, tgt, with_scaling=with_scaling
    ).transformation

    # Apply transformation to source
    ones = np.ones((source_points.shape[0], 1))
    source_hom = np.hstack((source_points, ones))  # (N, 4)
    aligned = (transformation @ source_hom.T).T[:, :3]

    return aligned, transformation



def ATE(gt_traj=None, traj2D=None, traj3D=None):
    
    if gt_traj is None or len(gt_traj) == 0:
        print("No ground truth to compare")
        return
    
    if (traj2D is None or len(traj2D) == 0) and (traj3D is None or len(traj3D) == 0):
        print("No trajectory as input")
        return

    gt_traj=np.array(gt_traj)

    if traj3D is not None:
        traj3D = np.array(traj3D)
        length = min(len(gt_traj), len(traj3D))
        aligned_3D= traj3D[:length]
        gt_traj= gt_traj[:length]
        errors_3D = np.linalg.norm(aligned_3D - gt_traj[:length], axis=1)
        rmse_3D = np.sqrt(np.mean(errors_3D ** 2))
        print(f"[ATE] VO 3D RMSE: {rmse_3D:.4f} m")
    
    if traj2D is not None:
        traj2D= np.array(traj2D)
        length= min(len(traj2D), len(gt_traj))
        aligned_2D= traj2D[:length]
        gt_traj= gt_traj[:length]
        errors_2D = np.linalg.norm(aligned_2D - gt_traj[:length], axis=1)
        rmse_2D = np.sqrt(np.mean(errors_2D ** 2))
        print(f"[ATE] VO 2D RMSE: {rmse_2D:.4f} m")

