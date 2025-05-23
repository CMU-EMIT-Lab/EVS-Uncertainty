# Statistical Analysis to Assess Part-to-Coupon Porosity Equivalence for Fatigue of Additively Manufactured Parts

## Installing

Clone the repository on your local machine
```shell
git clone https://github.com/CMU-EMIT-Lab/EVS-Uncertainty.git
```

## Data Labeling/Reading

To use the test data that was used in the referenced paper, download it from this link https://doi.org/10.1184/R1/26304175 (or use downloader.py). To refit thresholds, run threshold_choice.py on this data. To obtain CDFs from simulation, run boostraploop.py. To adapt to your own data, you can create your own poredf of data and parameters.csv with the relevant columns.


## Reference

Please use the following reference if you utilize this code.

```
@article{miner2025stats,
  title={Extreme value statistics with uncertainty to assess porosity equivalence across additively
manufactured parts},
  author={Miner, Justin P. and Narra, Sneha Prabha},
  journal={Reliability Engineering \& System Safety},
  volume={262},
  pages={111207},
  year={2025},
  publisher={Elsevier},
  doi = {10.1016/j.ress.2025.111207}
}
