# noises sorted by characteristics
# see notebooks/experiment_perceptually_weighted.ipynb

thrs_std = 200      # threshold for stationary
thrs_mean = 1500    # threshold for narrowband

_pinknoise_gain = 3.5
_rwnoises = {
 'DKITCHEN': {
  'gain': 4.622552871704102,
  'bw_std': 163.4586452875232,
  'bw_mean': 2772.925660063322,
  'filepath': '/data/riccardo_datasets/demand/DKITCHEN/ch01.wav'},
 'DLIVING': {
  'gain': 4.948243618011475,
  'bw_std': 341.72295151689934,
  'bw_mean': 1565.415095225999,
  'filepath': '/data/riccardo_datasets/demand/DLIVING/ch01.wav'},
 'DWASHING': {
  'gain': 14.610499382019043,
  'bw_std': 330.5283161039559,
  'bw_mean': 1397.3861501102917,
  'filepath': '/data/riccardo_datasets/demand/DWASHING/ch01.wav'},
 'NFIELD': {
  'gain': 11.365270614624023,
  'bw_std': 120.61892705476156,
  'bw_mean': 1268.089690197702,
  'filepath': '/data/riccardo_datasets/demand/NFIELD/ch01.wav'},
 'NPARK': {
  'gain': 4.841303825378418,
  'bw_std': 214.36562023947923,
  'bw_mean': 1803.2941239146132,
  'filepath': '/data/riccardo_datasets/demand/NPARK/ch01.wav'},
 'NRIVER': {
  'gain': 6.999195098876953,
  'bw_std': 159.83276169668696,
  'bw_mean': 1772.1916193437792,
  'filepath': '/data/riccardo_datasets/demand/NRIVER/ch01.wav'},
 'OHALLWAY': {
  'gain': 9.420412063598633,
  'bw_std': 181.44414637922964,
  'bw_mean': 1453.27783326968,
  'filepath': '/data/riccardo_datasets/demand/OHALLWAY/ch01.wav'},
 'OMEETING': {
  'gain': 2.5379886627197266,
  'bw_std': 319.8988305731093,
  'bw_mean': 1563.4266276314186,
  'filepath': '/data/riccardo_datasets/demand/OMEETING/ch01.wav'},
 'OOFFICE': {
  'gain': 13.414482116699219,
  'bw_std': 251.8369742079215,
  'bw_mean': 1501.0606541358102,
  'filepath': '/data/riccardo_datasets/demand/OOFFICE/ch01.wav'},
 'PCAFETER': {
  'gain': 3.4222795963287354,
  'bw_std': 259.30289605120464,
  'bw_mean': 1810.4410461981229,
  'filepath': '/data/riccardo_datasets/demand/PCAFETER/ch01.wav'},
 'PRESTO': {
  'gain': 1.5587252378463745,
  'bw_std': 186.1938661738654,
  'bw_mean': 2075.3030426613072,
  'filepath': '/data/riccardo_datasets/demand/PRESTO/ch01.wav'},
 'PSTATION': {
  'gain': 3.9170989990234375,
  'bw_std': 114.46872347812172,
  'bw_mean': 1681.9209442703452,
  'filepath': '/data/riccardo_datasets/demand/PSTATION/ch01.wav'},
 'SCAFE': {
  'gain': 4.958830833435059,
  'bw_std': 212.18781042615367,
  'bw_mean': 1746.6234981875018,
  'filepath': '/data/riccardo_datasets/demand/SCAFE/ch01.wav'},
 'SPSQUARE': {
  'gain': 7.396984100341797,
  'bw_std': 201.8272149875209,
  'bw_mean': 1565.9187743804325,
  'filepath': '/data/riccardo_datasets/demand/SPSQUARE/ch01.wav'},
 'STRAFFIC': {
  'gain': 5.732484817504883,
  'bw_std': 228.18287954611145,
  'bw_mean': 1844.0874355423523,
  'filepath': '/data/riccardo_datasets/demand/STRAFFIC/ch01.wav'},
 'TBUS': {
  'gain': 13.016814231872559,
  'bw_std': 338.59387859252683,
  'bw_mean': 1547.364284586548,
  'filepath': '/data/riccardo_datasets/demand/TBUS/ch01.wav'},
 'TCAR': {
  'gain': 15.270461082458496,
  'bw_std': 174.25274988125037,
  'bw_mean': 654.8313490697777,
  'filepath': '/data/riccardo_datasets/demand/TCAR/ch01.wav'},
 'TMETRO': {
  'gain': 5.786309242248535,
  'bw_std': 224.31585757158888,
  'bw_mean': 1701.8706205143699,
  'filepath': '/data/riccardo_datasets/demand/TMETRO/ch01.wav'}}


# function to get noises
def get_rwnoises(stationary='both', narrowband='both'):
    noises = {}
    # filter by stationarity (bw std)
    for i in _rwnoises:
        if stationary == 'both':
            noises[i] = _rwnoises[i]
        elif stationary == True and _rwnoises[i]['bw_std'] < thrs_std:
            noises[i] = _rwnoises[i]
        elif stationary == False and _rwnoises[i]['bw_std'] > thrs_std:
            noises[i] = _rwnoises[i]

    noises2 = {}
    # filter by bandwidth (bw mean)
    for i in noises:
        if narrowband == 'both':
            noises2[i] = _rwnoises[i]
        elif narrowband == True and _rwnoises[i]['bw_mean'] < thrs_mean:
            noises2[i] = _rwnoises[i]
        elif narrowband == False and _rwnoises[i]['bw_mean'] > thrs_mean:
            noises2[i] = _rwnoises[i]

    # extract useful infos only (and apply pink noise gain)
    noises_list = [{'filepath': noises2[i]['filepath'],
                    'gain': noises2[i]['gain'] - _pinknoise_gain} for i in noises2]
    return noises_list
