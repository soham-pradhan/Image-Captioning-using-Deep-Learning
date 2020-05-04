<!DOCTYPE HTML>
<html lang="en" >
<head>
	<meta charset="UTF-8">
	<title></title>
	<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.min.css">
	<link rel="stylesheet" href="style.css">

</head>
<body>
<div class="navbar">
  					<a href="http://localhost:8080/SITE/index.php">Home</a>
					
</div>
<style>
img {
    height: auto;
	width: 30%;
	padding-top: 150px;
}
</style>
<?php
$name = $_GET['image'];
$id = $_GET['id'];
echo "$name";
if($id==0)
{
	echo "<center><img src='caption_images/$name'></center>";
	echo "<br>";
	echo "<br>";

	exec("C:\Users\Ashish\Anaconda3\python.exe C:/xampp/htdocs/SITE/display.py C:/xampp/htdocs/SITE/caption_images/$name 2>&1", $output);
}
else
{
	echo "<center><img src='uploaded_images/$name'></center>";
	echo "<br>";
	echo "<br>";

	exec("C:\Users\Ashish\Anaconda3\python.exe C:/xampp/htdocs/SITE/display.py C:/xampp/htdocs/SITE/uploaded_images/$name 2>&1", $output);
}

foreach($output as $indx=>$caption)
{
	$output[$indx]=str_replace(">","","$caption");
}
$c=1;
foreach($output as $indx=>$caption)
{
	if($c==1)
	{
	echo "<center><p style='color:white'>Greedy Search: $caption</p></center>";
	echo "<br>";
	$c=$c+1;
	}
	else if($c==2)
	{
	echo "<center><p style='color:white'>Beam Search 3: $caption</p></center>";
	echo "<br>";
	$c=$c+1;
	}
	else if($c==3)
	{
	echo "<center><p style='color:white'>Beam Search 5: $caption</p></center>";
	echo "<br>";
	$c=$c+1;
	}
	else if($c==4)
	{
	echo "<center><p style='color:white'>Beam Search 7: $caption</p></center>";
	echo "<br>";
	$c=$c+1;
	}
}

?>
</body>
</html>