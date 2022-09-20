from Classification.Training.training import training

dataset = r"DATASETCIAT"
model_dir = "model_general1.h5"
lbldic = {
		"cultivo":0,
		"surco":1,
	}
trainer = training(path_to_dataset=dataset, 
	save_to_dir_model=model_dir, 
	lbl=lbldic, 
	bs=16, 
	epochs=50, 
	lr=0.001, 
	seed=60, 
	numclasses=len(lbldic), 
	sizeofimage=3, 
	splitDataset=0.2)

trainer.train()