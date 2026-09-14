# PathTokenSurv v1.7.0 Protocol

## Design
- 5 ensemble members per outer fold
- identical preprocessing
- identical splits
- identical architecture
- different stochastic initialization seeds

## Prediction
S_ens(t|x)=1/M sum_m S_m(t|x)

## Uncertainty
Variance across ensemble survival predictions.