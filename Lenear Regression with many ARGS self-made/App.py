
#Predict Price
def main():
    print('Self Made Linear Regression')
    file = open(r'train.tsv')
    file2 = open(r'in.tsv')

    X_price = []
    X_mileage = []
    X_year = []
    X_brand = []
    X_engineType = []
    X_engineCapacity = []
    carsX = []

    X_mileage_test = []
    X_year_test = []
    X_brand_test = []
    X_engineType_test = []
    X_engineCapacity_test = []
    carsX_test = []

    for line in file:
        line = line.split()

        X_price.append(line[0])
        X_mileage.append(line[1])
        X_year.append(line[2])
        X_brand.append(line[3])
        X_engineType.append(line[4])
        X_engineCapacity.append(line[5])

    carsX.append(X_price)
    carsX.append(X_mileage)
    carsX.append(X_year)
    carsX.append(X_brand)
    carsX.append(X_engineType)
    carsX.append(X_engineCapacity)

    for line in file2:
        line = line.split()

        X_mileage_test.append(line[0])
        X_year_test.append(line[1])
        X_brand_test.append(line[2])
        X_engineType_test.append(line[3])
        X_engineCapacity_test.append(line[4])

    carsX_test.append(X_mileage_test)
    carsX_test.append(X_year_test)
    carsX_test.append(X_brand_test)
    carsX_test.append(X_engineType_test)
    carsX_test.append(X_engineCapacity_test)



if __name__ == "__main__":
    main()