import time
from modules.Detect import Detect

# added
import uvicorn, asyncio, cv2
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer

web = WebGear(logging=True)#

start_time = time.time()
detect = Detect(256).start()

web.config["generator"] = detect.my_frame_producer#
uvicorn.run(web(), host="localhost", port=8000)#




print('Init finished')
print("--- took %s seconds ---" % (time.time() - start_time))
print()
web.shutdown()#
detect.enable()
time.sleep(1000)
