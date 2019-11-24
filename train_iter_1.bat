chcp 65001
:: #epochs: 150, Adam(lr=1e-4)
venv\Scripts\python.exe train.py --train 1 --num-epochs 150 --optimizer Adam --dataset-path "C:\Users\Sebastião Pamplona\Desktop\DEV\datasets\treated\age\wiki_aligned_uni_160\

:: #epochs: 150, SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
venv\Scripts\python.exe train.py --train 1 --num-epochs 150 --optimizer SGD --dataset-path "C:\Users\Sebastião Pamplona\Desktop\DEV\datasets\treated\age\wiki_aligned_uni_160\

:: #epochs: 150, RMSprop(lr=1e-4, rho=0.9)
venv\Scripts\python.exe train.py --train 1 --num-epochs 150 --optimizer RMSprop --dataset-path "C:\Users\Sebastião Pamplona\Desktop\DEV\datasets\treated\age\wiki_aligned_uni_160\


:: #epochs: 150, Adam(lr=1e-4)
venv\Scripts\python.exe train.py --train 1 --num-epochs 150 --optimizer Adam --dataset-path "C:\Users\Sebastião Pamplona\Desktop\DEV\datasets\treated\age\wiki_augmented_uni_160\

:: #epochs: 150, SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
venv\Scripts\python.exe train.py --train 1 --num-epochs 150 --optimizer SGD --dataset-path "C:\Users\Sebastião Pamplona\Desktop\DEV\datasets\treated\age\wiki_augmented_uni_160\

:: #epochs: 150, RMSprop(lr=1e-4, rho=0.9)
venv\Scripts\python.exe train.py --train 1 --num-epochs 150 --optimizer RMSprop --dataset-path "C:\Users\Sebastião Pamplona\Desktop\DEV\datasets\treated\age\wiki_augmented_uni_160\


