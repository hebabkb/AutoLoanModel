
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_approval_matrix(y_pred, bad_flag, model):
    fig, ax = plt.subplots(figsize=(5, 5))

       # Compute the performance matrix (counts)
    perf_matrix = np.array([
        [sum((y_pred == 1) & (bad_flag == 1)), sum((y_pred == 1) & (bad_flag == 0))],  # Approved
        [sum((y_pred == 0) & (bad_flag == 1)), sum((y_pred == 0) & (bad_flag == 0))]   # Declined
    ])
    
    #calculate percentages
    total_approved = sum(y_pred == 1)
    total_declined = sum(y_pred == 0)
    
    perf_matrix_pct = np.zeros((2, 2))
    row_sums = perf_matrix.sum(axis=1)
    for i in range(2):
        perf_matrix_pct[i,:] = (perf_matrix[i,:] / row_sums[i] * 100)

    

     # Plot heatmap
    sns.heatmap(perf_matrix_pct, annot=True, fmt=".1f", cmap="coolwarm",
                xticklabels=['Bad Loan', 'Good Loan'],
                yticklabels=['Approved', 'Declined'],
                ax=ax)

    ax.set_title(f'Approval Decision Analysis (% by decision): {model_name}')
    ax.set_xlabel('Actual Loan Performance')
    ax.set_ylabel('Model Decision')

    plt.tight_layout()
    plt.show()
