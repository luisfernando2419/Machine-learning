namespace MachineLearning

//El código en C# utiliza **ML.NET** para construir un modelo de clasificación que diagnostica enfermedades a partir de síntomas.
//Primero, define las clases para representar los datos de entrada (síntomas) y salida (diagnóstico).
//Luego, crea un conjunto de datos de ejemplo que se carga y se usa para entrenar un modelo con un clasificador de máxima entropía,
//adecuado para clasificaciones multiclase. Con el modelo entrenado, el sistema puede predecir el diagnóstico de un nuevo paciente
//en función de sus síntomas y mostrar el resultado en la consola. Este proceso permite clasificar condiciones médicas básicas según los síntomas proporcionados.
{
    using System;
    using Microsoft.ML;
    using Microsoft.ML.Data;

    // Clase de entrada que representa los síntomas

    public class SymptomData
    {
        [LoadColumn(0)] public float Fiebre { get; set; }
        [LoadColumn(1)] public float Tos { get; set; }
        [LoadColumn(2)] public float DolorGarganta { get; set; }
        [LoadColumn(3)] public float DolorCabeza { get; set; }
        [LoadColumn(4)] public string Diagnostico { get; set; }
    }

    // Clase de salida que representa el resultado del modelo
    public class SymptomPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedDiagnostico { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Crear contexto de ML
            MLContext mlContext = new MLContext();

            // Crear el conjunto de datos de ejemplo
            var trainingData = new[]
            {
            new SymptomData { Fiebre = 1, Tos = 1, DolorGarganta = 1, DolorCabeza = 0, Diagnostico = "Gripe" },
            new SymptomData { Fiebre = 0, Tos = 1, DolorGarganta = 1, DolorCabeza = 0, Diagnostico = "Resfriado" },
            new SymptomData { Fiebre = 1, Tos = 0, DolorGarganta = 1, DolorCabeza = 1, Diagnostico = "Gripe" },
            new SymptomData { Fiebre = 1, Tos = 1, DolorGarganta = 0, DolorCabeza = 1, Diagnostico = "Gripe" },
            new SymptomData { Fiebre = 0, Tos = 1, DolorGarganta = 0, DolorCabeza = 1, Diagnostico = "Resfriado" },
            new SymptomData { Fiebre = 0, Tos = 0, DolorGarganta = 1, DolorCabeza = 1, Diagnostico = "Resfriado" }
        };

            // Cargar los datos en formato IDataView
            IDataView data = mlContext.Data.LoadFromEnumerable(trainingData);

            // Definir el pipeline de entrenamiento
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Diagnostico")
                .Append(mlContext.Transforms.Concatenate("Features", "Fiebre", "Tos", "DolorGarganta", "DolorCabeza"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Entrenar el modelo
            var model = pipeline.Fit(data);

            // Crear un nuevo ejemplo de datos
            var nuevoPaciente = new SymptomData { Fiebre = 1, Tos = 1, DolorGarganta = 0, DolorCabeza = 1 };

            // Crear el predictor
            var predictor = mlContext.Model.CreatePredictionEngine<SymptomData, SymptomPrediction>(model);

            // Hacer la predicción
            var prediccion = predictor.Predict(nuevoPaciente);

            // Mostrar el resultado
            Console.WriteLine($"Diagnóstico para el nuevo paciente: {prediccion.PredictedDiagnostico}");
        }
    }

}
