# YOLO-API
REST API to use vanilla yolov3 darknet model online


Use the following command to use the API:</br>
```curl.exe -X POST -F image=@imagename.jpg "https://yolo-rest-api.herokuapp.com/api"```

Return value : List of (x,y,w,h,label,confidence)

(x,y) = Top left coordinates of the bounding box of the image</br>
(w,h) = width and height of the bounding box</br>
label = name of the object</br>
confidence = confidence of the prediction</br>

### TODO:
- Optimize the API to enable videos/livestreams as inputs</br>
- Deploy on GPU</br>
- Add user authentication</br>
