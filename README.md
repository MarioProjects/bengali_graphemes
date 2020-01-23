Hello world!

|   Features   |     Head    |   Size   |  DA  |      More     |            Local           |   LB   |
|:------------:|:-----------:|:--------:|:----:|:-------------:|:--------------------------:|:------:|
| Densenet121  | InitialHead | 150->128 | da7  | ------------- | 0.968 -> 0.956 0.985 0.975 | ------ |
| SeresNext101 | InitialHead | 150->128 | da7  | ------------- | 0.976 -> 0.966 0.989 0.984 | ------ |
| SeresNext101 | InitialHead | 150->128 | da7  | extendedTrain | 0.976 -> 0.966 0.989 0.983 | ------ |
| SeresNext101 | InitialHead | 150->128 | da8  | extendedTrain | 0.977 -> 0.967 0.989 0.984 | 0.9659 |
| SeresNext101 | InitialHead |   128    | da7b | ClipGrad 1.0  | 0.977 -> 0.967 0.988 0.984 | ------ |