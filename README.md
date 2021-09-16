# DropoutNet
This is the pytorch implementation of **NeurIPS'17 DropoutNet: Addressing Cold Start in Recommender Systems** 

To see paper, use this [Link](http://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf)


## Environment Requirement
`pip3 install -r requirements.txt`

## Usage
1. download the data
    1. use this [link](https://s3.amazonaws.com/public.layer6.ai/DropoutNet/recsys2017.pub.tar.gz) for downloading recsys 2017 dataset.
    2. locate data in `./data`
2. run `python3 main.py run`
3. if you want to run it with [supervisord](http://supervisord.org/), run `supervisord -c src/conf/spv.conf`

## Custom
you can customize options in `./src/conf/config.json`.
