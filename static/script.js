let canvas = document.getElementById('myCanvas'),
    context = canvas.getContext('2d'),
    w = canvas.width,
    h = canvas.height;

let mouse = {x:0, y:0}
let draw = false;

context.lineWidth = 20;


canvas.addEventListener('mousedown', function(e){
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
    draw = true;
    context.beginPath();
    context.moveTo(mouse.x, mouse.y);
})

canvas.addEventListener('mousemove', function(e){
    if (draw == true)
    {
        mouse.x = e.pageX - this.offsetLeft;
        mouse.y = e.pageY - this.offsetTop;   
        context.lineTo(mouse.x, mouse.y);
        context.strokeStyle='white';
        context.stroke();
    }
})

canvas.addEventListener('mouseup', function(e){
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
    context.lineTo(mouse.x, mouse.y);
    context.closePath();
    draw = false;
})


function clearCanv()
{
    context.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predict').textContent = ''
}


function save()
{
    var filename = 'pic';
    var image = canvas.toDataURL();
    $.post("/", { save_fname: filename, save_image: image })
    .done(function(data){
        document.getElementById('predict').textContent = 'Вы нарисовали ' + data['predict']
    });
}



// function getImage(canvas){
//     let imageData = canvas.toDataURL();
//     let image = new Image();
//     image.src = imageData;
//     image.width = '28';
//     image.height = '28';
//     image.name = pic;
//     document.getElementById('main').appendChild(image);
//     return image;
// }

// function saveImage(image) {
//     let link = document.createElement("a");
 
//     link.setAttribute("href", image.src);
//     link.setAttribute("download", "canvasImage");
//     link.click();
// }

// function saveCanvasAsImageFile(){
//     var image = getImage(canvas);
//     saveImage(image);
// }


