<!DOCTYPE HTML>
<html lang="en" >
<head>
  <meta charset="UTF-8">
  <title>Image Captioning</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.min.css">
  
<link rel="stylesheet" href="style.css">
<? php
	session_start();
?>
</head>
<body>
<style>
	.inputfile {
	width: 0.1px;
	height: 0.1px;
	opacity: 0;
	overflow: hidden;
	position: absolute;
	z-index: -1;
	
}
 .button{
  background-color: #4CAF50; /* Green */
	
  border: none;
  color: white;
  padding: 16px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  
  transition-duration: 0.4s;
  cursor: pointer;
  
  border-radius: 10px
  
  
}
.button2 {
	
  background-color: #2ab1ce;	
  color:  white;
  border: 5px solid #2ab1ce;
  margin-right:550px;
}

.button2:hover {
  background-color: white; 
  
  color: black;
}
.button3 {
	float:left;
	margin-left:595px;
	
  background-color: #f44336; 
  color: white; 
  border: 5px solid #f44336;
}

.button3:hover {
  background-color: white;

  color: black;
}

</style>
<div class="carousel right">
  <div class="slide"></div>
  <div class="wrap">
    <ul>
      <?php
		$dir='learn_images';
		$files1 = scandir($dir);
		#print_r($files1);
		$count=0;
		unset($files1[0]);
		unset($files1[1]);
		shuffle($files1);
		
		foreach($files1 as $indx=>$name)
		{
			$count=$count+1;
			if($count<=5){
				echo "<li><a href='identify.php?image=$name&id=0'><img src='learn_images/$name' alt=''/></a></li>";
			}
		}
		
	  ?>

    </ul>
  </div>
  
</div>
<div class="navbar">
  					<a href="http://localhost:8080/SITE/index.php">Home</a>
					<a href="http://localhost:8080/SITE/learn.php">Learn</a>
</div>
<script src='http://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js'></script>
<script  src="function.js"></script>
<div style="padding-top:580px">
<form method="post" enctype="multipart/form-data">
    <input type="file" name="fileToUpload" id="fileToUpload" class="inputfile">
	<center><label for="fileToUpload" class="button button3">Select Image</label><center>
	
    <center><input type="submit" value="Upload Image" name="submit" class="button button2"></center>
	<div id="file-upload-filename" style="color:white; padding:20px"></div>
</form>
<script>
  var input = document.getElementById( 'fileToUpload' );
  var infoArea = document.getElementById( 'file-upload-filename' );

  input.addEventListener( 'change', showFileName );

  function showFileName( event ) {

    // the change event gives us the input it occurred in 
    var input = event.srcElement;

    // the input has an array of files in the `files` property, each one has a name that you can use. We're just using the name here.
    var fileName = input.files[0].name;

    // use fileName however fits your app best, i.e. add it into a div
    infoArea.textContent = 'File name: ' + fileName;
  }
</script>
 </div>
</body>
</body>
</html>
<?php
if(isset($_POST['submit']))
{
$target_dir = "uploaded_images/";
$target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
$_SESSION["name"] = basename($_FILES["fileToUpload"]["name"]);
$name = $_SESSION["name"];
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
// Check if image file is a actual image or fake image

// Check if file already exists
		
// Allow certain file formats
if($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg" && $imageFileType != "jfif") {
    echo "<script>alert('Sorry, only JPG, JPEG, JFIF and PNG files are allowed.')</script>";
    $uploadOk = 0;
}
// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 0) {
    echo "<script>alert('Sorry, your file was not uploaded.')</script>";
// if everything is ok, try to upload file
} else {
    if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file)) {
        echo "<script>alert('The file has been uploaded successfully.');window.location.href='identify.php?image=$name&id=1';</script>";

    } else {
        echo "<script>alert('Sorry, there was an error uploading your file.')</script>";
    }
}
}
?>