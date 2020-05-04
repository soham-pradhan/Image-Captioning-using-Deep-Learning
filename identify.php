<!DOCTYPE HTML>
<html lang="en" >
<head>
	<meta charset="UTF-8">
	<title></title>
	<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.min.css">
	<link rel="stylesheet" href="style.css">

</head>
<body>
<style>
img {
    height: auto;
	width: 30%;
	padding-top: 150px;
}
</style>
<div class="navbar">
  					<a href="http://localhost:8080/SITE/learn.php">Home</a>
					
</div>
<?php
$name = $_GET['image'];
$id = $_GET['id'];
echo "$name";

if($id==0)
{
	echo "<center><img src='learn_images/$name'></center>";
	echo "<br>";
	echo "<br>";
	
	exec("C:/Users/Ashish/Anaconda3/envs/env/python.exe C:/xampp/htdocs/SITE/classify.py C:/xampp/htdocs/SITE/learn_images/$name 2>&1", $output);
	
}
else
{
	echo "<center><img src='uploaded_images/$name'></center>";
	echo "<br>";
	echo "<br>";

	exec("C:/Users/Ashish/Anaconda3/envs/env/python.exe C:/xampp/htdocs/SITE/classify.py C:/xampp/htdocs/SITE/uploaded_images/$name 2>&1", $output);
}

foreach($output as $indx=>$caption)
{
	$output[$indx]=str_replace(">","","$caption");
}
foreach($output as $indx=>$caption)
{
	echo "<center><p style='color:white'>$caption</p></center>";
	echo "<br>";
}

?>
</body>
</html>