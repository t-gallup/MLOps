from metaflow import FlowSpec, step, Parameter, current

class ModelTrainingFlow(FlowSpec):
    # Define parameters
    cv_folds = Parameter('cv_folds', default=5)
    experiment_name = Parameter('experiment_name', default='lab-6-metaflow')
    random_seed = Parameter('random_seed', default=42)

    @step
    def start(self):
        import mlflow
        import os
        # Set up MLFlow
        # os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:8080"
        mlflow.set_tracking_uri("http://localhost:8080")
        mlflow.set_experiment(self.experiment_name)

        self.next(self.load_data)

    @step
    def load_data(self):
        import dataprocessing
        import pickle
        print("Loading Data")
        # Load and process data
        self.data = dataprocessing.load_data()
        self.data = dataprocessing.clean_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = dataprocessing.split_data(self.data)
        
        with open('X_test.pkl', 'wb') as f:
            pickle.dump(self.X_test, f)
        
        with open('y_test.pkl', 'wb') as f:
            pickle.dump(self.y_test, f)
            
        self.next(self.train_model)

    @step
    def train_model(self):
        import modeltraining
        # Train models and get the best one
        print("Training Models")
        self.best_model, self.best_metrics = modeltraining.train_with_tuning(
            self.X_train,
            self.y_train,
            cv_folds=self.cv_folds,
            random_seed=self.random_seed
        )

        self.next(self.register_model)

    @step
    def register_model(self):
        import mlflow
        print("Registering Best Model")
        mlflow.set_tracking_uri("http://localhost:8080")
        with mlflow.start_run():
            # Log all parameters, metrics, and model
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("random_seed", self.random_seed)
            for metric_name, metric_value in self.best_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
            model_info = mlflow.sklearn.log_model(
                self.best_model,
                "model",
                registered_model_name="best_model"
            )
            # Register the model
            mlflow.register_model(
                model_uri=model_info.model_uri,
                name="my_best_model"
            )
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed successfully!")
        print(f"Best model metrics: {self.best_metrics}")

if __name__ == '__main__':
    ModelTrainingFlow()
