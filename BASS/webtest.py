# import necessary libs
import uvicorn, asyncio, cv2
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer

# initialize WebGear app without any source
web = WebGear(logging=True)

# create your own custom frame producer
async def my_frame_producer():

    # !!! define your own video source here !!!
    # Open any video stream such as live webcam 
    # video stream on first index(i.e. 0) device
    stream = cv2.VideoCapture(0)
    # loop over frames
    while True:
        # read frame from provided source
        (grabbed, frame) = stream.read()
        # break if NoneType
        if not grabbed:
            break

        # do something with your OpenCV frame here

        # reducer frames size if you want more performance otherwise comment this line
        frame = await reducer(frame, percentage=30)  # reduce frame by 30%
        # handle JPEG encoding
        encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
        # yield frame in byte format
        yield (b"--frame\r\nContent-Type:video/jpeg2000\r\n\r\n" + encodedImage + b"\r\n")
        await asyncio.sleep(0.00001)
    # close stream
    stream.release()


# add your custom frame producer to config
web.config["generator"] = my_frame_producer

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()