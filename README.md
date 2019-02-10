**You can start the code directly with `python3 main.py` if you installed numpy and matplotlib manually.**
**These steps are not necessarily needed for running the code.**
**Steps for running the code from nothing:**
1. Download the files and open this directory(the directory which includes this README file) in terminal
2. `sudo apt-get install python3-tk`   (Tkinter is a backend graphical engine, needed for matplotlib and also comes with mostly python packages but i need to install in ubuntu 18.04. **Skip if you already have a graphical engine for matplotlib.**
3. `pip install pipenv` Needed for installing numpy and matplotlib easily. 
4. `pipenv install` I've prepared a Pipfile for easily installing matplotlib and numpy. Use this command to create a virtual env and install them.
5. `pipenv run python3 main.py`

You can see the results in the pdf file named Report.pdf

Datasets used in this project:
* https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
* https://archive.ics.uci.edu/ml/datasets/Ionosphere

