FROM 10jqkaaicubes/cuda:11.0-py3.8.5

COPY ./ /home/jovyan/stock_market_deal

RUN cd /home/jovyan/stock_market_deal && \
    python -m pip install -r requirement.txt 