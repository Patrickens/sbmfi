param (
    [String]$path,
    [String]$image,
    [Int] $prepare = 0,
    [String] $conc_prior
)
$windows_path = $path
$file_name = Split-Path $windows_path -leaf
$file_no_ext = [io.path]::GetFileNameWithoutExtension($windows_path)

Write-Output $prepare

docker start $image
docker cp "C:\python_projects\pysumo\docker_pta\run_pta.py" "${image}:/run_pta.py"  # this is the script that we run on 'path'
docker cp $path "${image}:/$file_name"
$command = "python run_pta.py -v -f $file_name -prep $prepare "
if ($conc_prior.Length -gt 0){
    $command = $command + "-conc_prior $conc_prior"
    $file_no_ext = $file_no_ext + "_$conc_prior"
}
$file_name = "$file_no_ext" + "_tfs.p"
Write-Output $command
Write-Output $file_name
docker exec $image bash -c $command
docker cp ("${image}:$file_name") ("$file_name")
