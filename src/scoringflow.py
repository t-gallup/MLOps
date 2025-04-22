from metaflow import FlowSpec, step, Parameter

class ModelScoringFlow(FlowSpec):
    model_name = Parameter('model_name', default='best_model')
    model_stage = Parameter('model_stage', default='Production')
    X_path = Parameter('X_test_path', default='X_test.pkl')
    y_path = Parameter('y_test_path', default='y_test.pkl')

    @step
    def start(self):
        import mlflow
        mlflow.set_tracking_uri("http://localhost:8080")
        
        self.next(self.load_data)
        
    @step
    def load_data(self):
        import pickle
        print("Loading Data")
        # Load the test data saved during training
        with open(self.X_path, 'rb') as f:
            self.X_test = pickle.load(f)
        with open(self.y_path, 'rb') as f:
            self.y_test = pickle.load(f)
        
        self.next(self.load_model)
    
    @step
    def load_model(self):
        import mlflow
        # Load the model from MLFlow registry
        print("Loading Best Model")
        model_uri = f"models:/{self.model_name}/1"
        mlflow.set_tracking_uri("http://localhost:8080")
        self.model = mlflow.pyfunc.load_model(model_uri)

        self.next(self.score_data)
    
    @step
    def score_data(self):
        import pandas as pd
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        print("Predicting Test Set")
        # Make predictions
        self.predictions = self.model.predict(self.X_test)
        self.results = pd.DataFrame({
            'id': range(len(self.predictions)),
            'prediction': self.predictions
        })
        
        # Compute performance metrics
        true_values = self.y_test
        self.rmse = np.sqrt(mean_squared_error(true_values, self.predictions))
        self.r2 = r2_score(true_values, self.predictions)
        self.mae = mean_absolute_error(true_values, self.predictions)
        
        print(f"Test RMSE: {self.rmse:.4f}")
        print(f"Test R² Score: {self.r2:.4f}")
        print(f"Test MAE: {self.mae:.4f}")
        
        self.next(self.output_predictions)
    
    @step
    def output_predictions(self):
        # Output predictions to CSV
        print("Outputting Predictions")
        output_path = 'predictions.csv'
        self.results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow.
        """
        print("Scoring flow completed successfully!")
        print(f"Generated {len(self.predictions)} predictions")

if __name__ == '__main__':
    ModelScoringFlow()
