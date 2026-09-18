# LangChain Agent Basics

This repository is a learning project created to understand how AI agents work with LangChain, starting from the most basic concepts and gradually moving towards more advanced agent architectures.

The project is part of the preparation for my Bachelor's Thesis (TFG), where the final objective will be related to agentic AI applied to autonomous systems.

The idea of this repository is not only to make the code work, but to understand every step involved:

- how the development environment is configured,
- how Python environments work,
- how dependencies are installed,
- how API keys are managed securely,
- how Git and GitHub are used,
- how LangChain agents work,
- how agents use tools,
- and how all these concepts can later be applied to more complex systems.



# 0. First of all, I explain how another person can install this project.

Someone cloning this repository from GitHub would first need:

- Git
- Python 3.12 or a compatible Python version
- WSL/Ubuntu if working on Windows

Clone the repository: git clone <REPOSITORY_URL>

Enter the directory: cd langchain_agent_basics

Create a virtual environment: python3 -m venv .venv

Activate it: source .venv/bin/activate

Install all Python dependencies: pip install -r requirements.txt

Create the private environment file: touch .env

Use `.env.example` as reference.

The private `.env` must contain: OPENAI_API_KEY=<YOUR_REAL_OPENAI_API_KEY>

Verify that the key can be loaded: 
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OPENAI_API_KEY is set:', bool(os.getenv('OPENAI_API_KEY')))"

Expected: OPENAI_API_KEY is set: True

The project is then ready to run.


Below I will now give a detailed explanation of the operation and structure of the project:

# 1. Development environment

The main operating system of my computer is Windows.

However, the project is developed inside Ubuntu using WSL (Windows Subsystem for Linux).

The general structure is:

Windows
│
├── Visual Studio Code
│
│
│   connected to
│
└── WSL / Ubuntu
    │
    ├── Python
    ├── Git
    ├── Python virtual environments
    ├── LangChain
    └── Project files

This means that Visual Studio Code is still a Windows application, but the terminal, Python interpreter, Git commands and project files are running inside Ubuntu.

This setup is useful because many of the technologies that may be used later in the Bachelor's Thesis are commonly used in Linux environments, such as:

- Docker
- ArduPilot SITL
- MAVLink
- ROS
- LangChain
- LangGraph



# 2. Installing and opening WSL in Visual Studio Code

The project is stored inside the Ubuntu filesystem instead of the Windows Desktop.

The main folder used for the Bachelor's Thesis projects is: /home/joelm/TFG

In Ubuntu, the symbol:  ~
represents the user's home directory.

Therefore: ~/TFG
is equivalent to: /home/joelm/TFG

To create this directory: mkdir -p ~/TFG

Then enter the directory: cd ~/TFG

The current directory can always be checked with the command: pwd

If Visual Studio Code asks for permission to access:   wsl.localhost
the connection can be allowed because it corresponds to the local WSL environment.



# 3. Checking that Visual Studio Code is really using WSL

It is important that Visual Studio Code is not simply opening the Ubuntu folder as a Windows network folder. Visual Studio Code must be connected directly to WSL. 

The bottom-left corner of Visual Studio Code should display: WSL: Ubuntu

If it does not appear, open the remote connection menu in the bottom-left corner and select: WSL

Then reopen the project folder inside Ubuntu.
When everything is configured correctly, the architecture is:

Windows
    │
    ▼
Visual Studio Code
    │
    ▼
WSL: Ubuntu
    │
    ▼
/home/joelm/TFG



# 4. Checking Python and Git

Before creating the project, verify that Python is installed: python3 --version

Current version used in this project: Python 3.12.3

Verify that Git is installed: git --version

Current version: git version 2.43.0



# 5. Creating the project

From the main TFG directory: cd ~/TFG

Create the project folder (in my case): mkdir langchain_agent_basics

Enter the project: cd langchain_agent_basics



# 6. Creating a Python virtual environment

A Python virtual environment is used so that the packages installed for this project do not modify the global Python installation of Ubuntu.

The virtual environment is stored inside the project in a folder called: .venv

Create it with: python3 -m venv .venv


## Possible Ubuntu error

On a new Ubuntu installation, ir could appear an error like: The virtual environment was not created successfully because ensurepip is not available.

This happens because the Ubuntu package required to create Python virtual environments is not installed.

Install it with:
   sudo apt update
   sudo apt install python3.12-venv



# 7. Activating the virtual environment

Activate the virtual environment with: source .venv/bin/activate



# 8. Updating pip

`pip` is the Python package manager.

It is used to install libraries such as LangChain.

Update it inside the virtual environment: python -m pip install --upgrade pip

Check the installed version: pip --version

During the initial setup of this project, the version was: pip 26.2.1



# 9. Initial project files

The following files were created:
  touch README.md
  touch .gitignore
  touch .env.example

The project initially contains:
langchain_agent_basics/
│
├── .venv/
├── .env
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── 01_simple_agent.py



# 10. Protecting private files with .gitignore

The `.gitignore` file tells Git which files or directories must not be uploaded to the repository.

The current `.gitignore` contains:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

The meaning of each entry is: .env

Contains private environment variables such as the OpenAI API key: .venv/

Contains the local Python virtual environment and all installed packages: __pycache__/

Contains temporary files automatically generated by Python: *.pyc
Ignores compiled Python files.



# 11. Managing the OpenAI API key

The OpenAI API key must never be written directly inside the Python code.

It must also never be uploaded to GitHub.

The real key is stored in: .env



# 12. Loading the .env file in Python

Python does not automatically read `.env` files.

The package: python-dotenv
is used to load the variables contained inside `.env`.

Install it with: python -m pip install python-dotenv

A Python program can then load the file with: 

   from dotenv import load_dotenv
   load_dotenv()

Internally, the process is approximately:

.env
│
│ contains OPENAI_API_KEY
│
▼
load_dotenv()
│
▼
Python environment
│
▼
LangChain
│
▼
OpenAI API

The API key can be checked without displaying its actual value:

python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OPENAI_API_KEY is set:', bool(os.getenv('OPENAI_API_KEY')))"

Expected output: OPENAI_API_KEY is set: True

This verifies that Python can access the API key without exposing it.



# 13. Installing LangChain

The main packages required for this repository are:
  langchain
  langchain-openai

Install them inside the virtual environment: python -m pip install -U langchain langchain-openai



# 18. requirements.txt

Python projects normally contain a file called: requirements.txt

It stores the Python packages and versions used by the project. 

It was generated with: python -m pip freeze > requirements.txt

A new user cloning this repository can install exactly the same dependencies with: 
  pip install -r requirements.txt



# 19. First agent file

The first Python file for experimenting with LangChain agents was created with: 
  touch 01_simple_agent.py

The objective of this file will be to implement the first basic LangChain agent.

The first example will initially use:

one user
      │
      ▼
one LangChain agent
      │
      ▼
one language model
      │
      ▼
one simple tool

The objective is not only to execute the example.

The objective is to understand:

- what an agent is,
- what a model is,
- what a tool is,
- how LangChain connects them,
- how the model decides whether a tool is required,
- how arguments are generated for a tool,
- how the tool result returns to the model,
- and how the final answer is produced.



# 20. Learning roadmap

The project will progressively increase in complexity.

The planned learning path is approximately:

```text
1. Simple LangChain agent
        │
        ▼
2. One agent with one tool
        │
        ▼
3. One agent with several tools
        │
        ▼
4. Understanding tool calls
        │
        ▼
5. Structured outputs
        │
        ▼
6. Pydantic models
        │
        ▼
7. State and memory
        │
        ▼
8. LangGraph workflows
        │
        ▼
9. More advanced agents
        │
        ▼
10. Multi-agent systems
        │
        ▼
11. Agentic mission orchestration
        │
        ▼
12. Possible integration with UAV systems
```

The final purpose is to progressively connect the concepts learned here with the Bachelor's Thesis.

A future high-level architecture could eventually evolve towards something such as:

```text
User / Operator
      │
      ▼
Agentic AI
      │
      ├── Mission tools
      ├── Fleet-state tools
      ├── Validation tools
      └── Planning tools
      │
      ▼
Mission orchestration
      │
      ▼
UAV / UGV systems
```

This repository only covers the first learning stages.

More advanced experiments can later be separated into additional repositories so that each project has a clear purpose and the GitHub profile also shows the progression of the work.