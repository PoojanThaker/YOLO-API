# YOLO-API
REST API to use vanilla yolov3 online


Use the following command to use the API:
curl.exe -X POST -F image=@imagename.jpg 'https://yolo-rest-api.herokuapp.com/api'

Return value : List of (x,y,w,h,label,confidence)

(x,y) = Top left coordinates of the bounding box of the image
(w,h) = width and height of the bounding box
label = name of the object
confidence = confidence of the prediction

# TODO:
- Optimize the API to enable videos/livestreams as inputs
- Deploy on GPU
- Add user authentication
