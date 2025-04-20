from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, NaiveBayes, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import Row

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("DiseasePrediction").getOrCreate()

# Step 2: Load dataset from HDFS
file_path = "hdfs://localhost:9000/user/hadoop/disease_data/Disease_symptom_and_patient_profile_dataset.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# --- Show original (raw) data ---
print("Original Data:")
df.show()

# --- Preprocessing ---
# Step 3: Convert binary columns (Yes/No) to numeric (1/0)
df_preprocessed = df.withColumn("Fever", when(df["Fever"] == "Yes", 1).otherwise(0)) \
    .withColumn("Cough", when(df["Cough"] == "Yes", 1).otherwise(0)) \
    .withColumn("Fatigue", when(df["Fatigue"] == "Yes", 1).otherwise(0)) \
    .withColumn("Difficulty Breathing", when(df["Difficulty Breathing"] == "Yes", 1).otherwise(0)) \
    .withColumn("Gender", when(df["Gender"] == "Male", 1).otherwise(0)) \
    .withColumn("Outcome Variable", when(df["Outcome Variable"] == "Positive", 1).otherwise(0))

# Step 4: Convert categorical columns like Blood Pressure and Cholesterol to numeric
df_preprocessed = df_preprocessed.withColumn("Blood Pressure", when(df_preprocessed["Blood Pressure"] == "Low", 0)
                                             .when(df_preprocessed["Blood Pressure"] == "Normal", 1).otherwise(2)) \
    .withColumn("Cholesterol Level", when(df_preprocessed["Cholesterol Level"] == "Normal", 0).otherwise(1))

# Step 5: Handle missing values (drop rows with missing values)
df_preprocessed = df_preprocessed.na.drop()

# Step 6: Use MinMaxScaler to scale 'Age'
assembler = VectorAssembler(inputCols=["Age"], outputCol="Age_Vector")
df_preprocessed = assembler.transform(df_preprocessed)

scaler = MinMaxScaler(inputCol="Age_Vector", outputCol="Age_Scaled")
scaler_model = scaler.fit(df_preprocessed)
df_preprocessed = scaler_model.transform(df_preprocessed).drop("Age", "Age_Vector")

# --- Show preprocessed data ---
print("Preprocessed Data:")
df_preprocessed.show()

# Step 7: Assemble features into a single vector
feature_columns = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender", "Blood Pressure", "Cholesterol Level", "Age_Scaled"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_preprocessed = assembler.transform(df_preprocessed).select("features", "Outcome Variable")

# Step 8: Split dataset into training and testing sets
train, test = df_preprocessed.randomSplit([0.8, 0.2], seed=42)

# --- Model Building with Hyperparameter Tuning ---

# Random Forest Classifier with Hyperparameter Tuning
rf = RandomForestClassifier(labelCol="Outcome Variable", featuresCol="features")

paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

crossval_rf = CrossValidator(estimator=rf,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=BinaryClassificationEvaluator(labelCol="Outcome Variable"),
                             numFolds=3)

# Decision Tree Classifier with Hyperparameter Tuning
dt = DecisionTreeClassifier(labelCol="Outcome Variable", featuresCol="features")

paramGrid_dt = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .build()

crossval_dt = CrossValidator(estimator=dt,
                             estimatorParamMaps=paramGrid_dt,
                             evaluator=BinaryClassificationEvaluator(labelCol="Outcome Variable"),
                             numFolds=3)

# Naive Bayes (Direct usage without hyperparameter tuning)
nb = NaiveBayes(labelCol="Outcome Variable", featuresCol="features")

# Gradient Boosting Classifier with Hyperparameter Tuning
gbt = GBTClassifier(labelCol="Outcome Variable", featuresCol="features", maxIter=20)

paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 10]) \
    .build()

crossval_gbt = CrossValidator(estimator=gbt,
                              estimatorParamMaps=paramGrid_gbt,
                              evaluator=BinaryClassificationEvaluator(labelCol="Outcome Variable"),
                              numFolds=3)

# Step 9: Train the models
models = [crossval_rf, crossval_dt, nb, crossval_gbt]
model_names = ["Random Forest (Tuned)", "Decision Tree (Tuned)", "Naive Bayes", "Gradient Boosting (Tuned)"]

trained_models = []
for model, name in zip(models, model_names):
    print(f"Training {name}...")
    trained_model = model.fit(train)
    trained_models.append(trained_model)

    # Step 10: Predictions
    predictions = trained_model.transform(test)

    # Step 11: Evaluation Metrics
    evaluator = BinaryClassificationEvaluator(labelCol="Outcome Variable")

    # Accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Outcome Variable", metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions)

    # AUC
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

    # Precision, Recall, F1-Score
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="Outcome Variable", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="Outcome Variable", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="Outcome Variable", metricName="f1")

    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)

    # Confusion Matrix
    predictions.groupBy("Outcome Variable", "prediction").count().show()

    # Step 12: Print results
    print(
        f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Define disease mapping based on labels
disease_map = {
    0.0: "Influenza",
    1.0: "Common Cold",
    2.0: "Eczema",
    3.0: "Asthma",
    4.0: "Hyperthyroidism",
    5.0: "Allergic Rhinitis",
    6.0: "Anxiety Disorders",
    7.0: "Diabetes",
    8.0: "Gastroenteritis",
    9.0: "Pancreatitis",
    10.0: "Rheumatoid Arthritis",
    11.0: "Depression",
    12.0: "Liver Cancer",
    13.0: "Stroke",
    14.0: "Urinary Tract Infection",
    15.0: "Dengue Fever",
    16.0: "Hepatitis",
    17.0: "Kidney Cancer",
    18.0: "Migraine",
    19.0: "Muscular Dystrophy",
    20.0: "Sinusitis",
    21.0: "Ulcerative Colitis",
    22.0: "Bipolar Disorder",
    23.0: "Bronchitis",
    24.0: "Cerebral Palsy",
    25.0: "Colorectal Cancer",
    26.0: "Hypertensive Heart Disease",
    27.0: "Multiple Sclerosis",
    28.0: "Myocardial Infarction",
    29.0: "Urinary Tract Infection (UTI)",
    30.0: "Osteoporosis",
    31.0: "Pneumonia",
    32.0: "Atherosclerosis",
    33.0: "Chronic Obstructive Pulmonary Disease (COPD)",
    34.0: "Epilepsy",
    35.0: "Hypertension",
    36.0: "Obsessive-Compulsive Disorder (OCD)",
    37.0: "Psoriasis",
    38.0: "Rubella",
    39.0: "Cirrhosis",
    40.0: "Conjunctivitis (Pink Eye)",
    41.0: "Kidney Disease",
    42.0: "Malaria",
    43.0: "Spina Bifida",
    44.0: "Liver Disease",
    45.0: "Osteoarthritis",
    46.0: "Chickenpox",
    47.0: "Coronary Artery Disease",
    48.0: "Eating Disorders (Anorexia, Bulimia)",
    49.0: "Fibromyalgia",
    50.0: "Hemophilia",
    51.0: "Hypoglycemia",
    52.0: "Lymphoma",
    53.0: "Tuberculosis",
    54.0: "Klinefelter Syndrome",
    55.0: "Acne",
    56.0: "Brain Tumor",
    57.0: "Cystic Fibrosis",
    58.0: "Glaucoma",
    59.0: "Rabies",
    60.0: "Autism Spectrum Disorder (ASD)",
    61.0: "Crohn's Disease",
    62.0: "Hyperglycemia",
    63.0: "Melanoma",
    64.0: "Ovarian Cancer",
    65.0: "Turner Syndrome",
    66.0: "Zika Virus",
    67.0: "Cataracts",
    68.0: "Multiple Sclerosis",
    69.0: "Osteoporosis",
    70.0: "Pneumocystis Pneumonia (PCP)",
    71.0: "Scoliosis",
    72.0: "Sickle Cell Anemia",
    73.0: "Tetanus"
}

# Step 13: User Input for Prediction
print("\nEnter your symptoms and details for prediction:")
age = float(input("Age: "))
fever = int(input("Fever (1 for Yes, 0 for No): "))
cough = int(input("Cough (1 for Yes, 0 for No): "))
fatigue = int(input("Fatigue (1 for Yes, 0 for No): "))
difficulty_breathing = int(input("Difficulty Breathing (1 for Yes, 0 for No): "))
gender = int(input("Gender (1 for Male, 0 for Female): "))
blood_pressure = int(input("Blood Pressure (0 for Low, 1 for Normal, 2 for High): "))
cholesterol = int(input("Cholesterol Level (0 for Normal, 1 for High): "))

# Prepare input data for prediction
user_data = spark.createDataFrame(
    [(fever, cough, fatigue, difficulty_breathing, gender, blood_pressure, cholesterol, age)],
    ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender", "Blood Pressure", "Cholesterol Level", "Age"]
)

# Assemble and scale user input data
age_assembler = VectorAssembler(inputCols=["Age"], outputCol="Age_Vector")
user_age_vector = age_assembler.transform(user_data)
user_age_scaled = scaler_model.transform(user_age_vector)
user_data = user_data.drop("Age").join(user_age_scaled.select("Age_Scaled"), on=None)
user_data = assembler.transform(user_data).select("features")

# Step 14: Predictions on User Input
for trained_model, name in zip(trained_models, model_names):
    user_predictions = trained_model.transform(user_data)

    # Get prediction and probabilities
    result = user_predictions.select("features", "prediction", "probability").collect()[0]
    features = result['features']
    predicted_label = result['prediction']
    probabilities = result['probability']

    # Map prediction to the corresponding disease
    disease_name = disease_map.get(predicted_label, "Unknown Disease")
    print(f"\n{name} Model Prediction: {disease_name} (Probability: {probabilities})")

# Stop Spark session
spark.stop()
