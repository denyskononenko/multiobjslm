FROM intelliseqngs/ubuntu-minimal-20.04:3.0.5
WORKDIR /home/multiobjslm
RUN mkdir ./dat
COPY env.txt multiobj_slm.ipynb hgpr.py gpr_custom.py data_adapter.py  ./
COPY dat/LPBF_Vit105_dataset_ML.xlsx  dat/LPBF_Vit105_dataset_ML.xlsx
RUN pip3 install -r env.txt
CMD ["/bin/bash"]