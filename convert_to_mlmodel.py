import coremltools
from keras.preprocessing import load_model

model = load_model('/Users/LocNguyen/Desktop/DataSci/human-activity-recognition/wandb/run-20200528_015708-kdj0zy8l/model-best.h5')

coreml_model = coremltools.converters.keras.convert(model, input_names=['sensor_signals'], output_names=['output'])

print(coreml_model)
coreml_model.author = 'LocNguyen'
coreml_model.license = 'N/A'
coreml_model.short_description = 'Activity based recognition based on UCI dataset'
coreml_model.output_description['output'] = 'Probability of each activity'
coreml_model.output_description['classLabel'] = 'Labels of activity'

coreml_model.save('HARClassifier.mlmodel')
