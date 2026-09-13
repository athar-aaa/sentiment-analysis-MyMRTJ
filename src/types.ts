export type SentimentClass = 'Negatif' | 'Netral' | 'Positif';

export interface DatasetDistribution {
  Negatif: number;
  Netral: number;
  Positif: number;
}

export interface TrainingHistory {
  epoch: number[];
  train_loss: number[];
  eval_loss: number[];
  train_accuracy?: number[];
  eval_accuracy?: number[];
  train_f1?: number[];
  eval_f1?: number[];
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  'f1-score': number;
  support: number;
}

export interface ClassificationReport {
  Negatif: ClassMetrics;
  Netral: ClassMetrics;
  Positif: ClassMetrics;
  accuracy: number;
  'macro avg': {
    precision: number;
    recall: number;
    'f1-score': number;
    support: number;
  };
  'weighted avg': {
    precision: number;
    recall: number;
    'f1-score': number;
    support: number;
  };
}

export interface ModelMetrics {
  model_name: string;
  dataset_distribution: DatasetDistribution;
  labels: SentimentClass[];
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  classification_report: ClassificationReport;
  confusion_matrix: number[][];
  y_true: number[];
  y_pred: number[];
  training_history: TrainingHistory;
}
