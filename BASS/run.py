import time
from modules.Detect import Detect

start_time = time.time()

# int:Net size:256-512
# bool:Test from image
detect = Detect(256).start()

print('Init finished')
print("--- took %s seconds ---" % (time.time() - start_time))
print()
detect.enable()
time.sleep(1000)
