# global constants
categories = 4
batchSize = 100  # was 200
iterationsPerBatch = 100  # was 500  # make sure it's divisble by 'iterationDivider'
iterationDivider = 10  # for print control
imageSize = 16  # changing this means running the 'Preprocessing' script again if the DF's don't already exist!

# paths
savePath = './saved_variable'  # for saver object
train16DfPath = 'Data/dfTrain16.csv'
test16DfPath = 'Data/dfTest16.csv'
train64DfPath = 'Data/dfTrain64.csv'
test64DfPath = 'Data/dfTest64.csv'
export_dir = './model'  # for simple save fun's
