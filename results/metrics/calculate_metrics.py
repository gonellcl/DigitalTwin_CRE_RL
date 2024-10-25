import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


# Reinforcement Learning Metrics
def calculate_rl_metrics(rewards, episode_lengths):
    """
    Calculate RL-specific metrics like total reward, average episode length, and reward variance.
    """
    total_reward = np.sum(rewards)
    avg_reward = np.mean(rewards)
    reward_variance = np.var(rewards)
    avg_episode_length = np.mean(episode_lengths)

    return {
        'Total Reward': total_reward,
        'Average Reward': avg_reward,
        'Reward Variance': reward_variance,
        'Average Episode Length': avg_episode_length
    }


# Regression Metrics
def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics like MAE, RMSE, and R-squared.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R-Squared': r2
    }


# Classification Metrics
def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate classification metrics like accuracy, precision, recall, F1, and ROC-AUC.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        gini_coefficient = 2 * roc_auc - 1
        metrics['ROC-AUC'] = roc_auc
        metrics['Gini Coefficient'] = gini_coefficient

    return metrics


# Economic Metrics
def calculate_economic_metrics(occupied_space, total_space, total_rent, total_expenses):
    """
    Calculate economic metrics like occupancy rate, vacancy rate, NOI, and RevPAR.
    """
    occupancy_rate = occupied_space / total_space
    vacancy_rate = 1 - occupancy_rate
    noi = total_rent - total_expenses
    revpar = total_rent / total_space

    return {
        'Occupancy Rate': occupancy_rate,
        'Vacancy Rate': vacancy_rate,
        'Net Operating Income (NOI)': noi,
        'Revenue Per Available Square Foot (RevPAR)': revpar
    }


# Stability Metrics
def calculate_stability_metrics(reward_variance, convergence_rate):
    """
    Calculate stability metrics for the RL agent's performance.
    """
    return {
        'Reward Variance': reward_variance,
        'Convergence Rate': convergence_rate
    }


# Main Function to Calculate All Metrics
def calculate_all_metrics(y_true, y_pred, rewards, episode_lengths, occupied_space, total_space, total_rent,
                          total_expenses, y_prob=None):
    """
    Calculate all categories of metrics and return as a combined dictionary.
    """
    metrics = {}

    # RL Metrics
    rl_metrics = calculate_rl_metrics(rewards, episode_lengths)
    metrics.update(rl_metrics)

    # Regression Metrics
    regression_metrics = calculate_regression_metrics(y_true, y_pred)
    metrics.update(regression_metrics)

    # Classification Metrics (if applicable)
    if y_prob is not None:
        classification_metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
        metrics.update(classification_metrics)

    # Economic Metrics
    economic_metrics = calculate_economic_metrics(occupied_space, total_space, total_rent, total_expenses)
    metrics.update(economic_metrics)

    # Stability Metrics
    stability_metrics = calculate_stability_metrics(rl_metrics['Reward Variance'],
                                                    len(episode_lengths) / np.sum(episode_lengths))
    metrics.update(stability_metrics)

    return metrics
