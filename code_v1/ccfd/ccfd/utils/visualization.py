import optuna
import matplotlib.pyplot as plt

def plot_tuning_results(study):
    """Visualizes Optuna hyperparameter tuning results."""

    # Plot importance of hyperparameters
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.suptitle("Hyperparameter Importance")
    plt.savefig("hyperparameter_importance.png")
    plt.show()

    # Plot hyperparameter vs. accuracy
    fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    fig.suptitle("Hyperparameter Relationships")
    plt.savefig("hyperparameter_relationships.png")
    plt.show()
