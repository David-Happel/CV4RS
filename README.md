# CV4RS

Crop Classification from Sentinel-2 Time Series Data

```
python train.py --samples 5 --epochs 1 --batch_size 5 --no_process_data
```

## HPC

### git clone

```
git clone git@github.com:David-Happel/CV4RS.git
```

### Python

```
mkdir venv
mkdir venv/cv4rs
module load python/3.7.1
python3 -m venv /home/users/d/davidhappel/venv/cv4rs
source /home/users/d/davidhappel/venv/cv4rs/bin/activate

cd CV4RS
pip install -r requirements.txt
```

### load data

Mount

```
sshfs davidhappel@gateway.hpc.tu-berlin.de: clustermount/
```

then copy it over
