:: #epochs: 150, Adam(lr=1e-4)
venv\Scripts\python.exe train.py --train 1 --num-epochs 150 --optimizer Adam

:: #epochs: 150, SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
venv\Scripts\python.exe train.py --train 1 --num-epochs 150 --optimizer SGD

:: #epochs: 150, RMSprop(lr=1e-4, rho=0.9)
venv\Scripts\python.exe train.py --train 1 --num-epochs 150 --optimizer RMSprop

