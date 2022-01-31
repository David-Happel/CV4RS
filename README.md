# CV4RS

Crop Classification from Sentinel-2 Time Series Data

```
python train.py --samples 5 --epochs 2 --batch_size 5  --timepoints 6 --model bl --name run_name --no_process_data
```

## HPC

### Accessing server
Note: you have to be connected to the TUB network to access the HPC. Otherwise you have to use the TUB VPN. 

```
ssh <TUBID>@gateway.hpc.tu-berlin.de
```

### Downloading Repo
git clone

```
git clone git@github.com:David-Happel/CV4RS.git
```

### Python + virtual environment
Setting up the virtual environment on the server
Do this once and then it will be called in the bash script thereafter

```
mkdir venv
mkdir venv/cv4rs
module load python/3.7.1
python3 -m venv /home/users/d/davidhappel/venv/cv4rs
source /home/users/d/davidhappel/venv/cv4rs/bin/activate
```

### Installing requirements 

```
cd CV4RS
pip install -r requirements.txt
```

### Tranferring Data
Mount the HPC to your local file directory when everything else is setup

Note: transferring takes a long time on home wifi. Would recommend using the TUB network

```
sshfs davidhappel@gateway.hpc.tu-berlin.de: <filepath to where you want to access the folder>
```

OOnce mounted you can then copy over data in your file explorer/finder
