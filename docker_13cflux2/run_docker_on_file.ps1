param ($path, $image, [Int] $nconfig)
$windows_path = $path
$file_name = Split-Path $windows_path -leaf
$file_no_ext = [io.path]::GetFileNameWithoutExtension($windows_path)

docker start $image
docker cp $path "${image}:/input/$file_name"
docker cp "run_pta.py" "${image}:/input/$file_name"
docker exec $image bash -c "USER=nzamboni fmlsign -i /input/$file_name -o /signed/signed_$file_name" # exec -d if you want to suppress output

for ($num = 0; $num -le $nconfig - 1; $num++){
    $signed_file = "/signed/signed_$file_name"
    $h5_file = "/output/fwdsim_${file_no_ext}_${num}.h5"
    $fwdsim_file = "/output/fwdsim_${file_no_ext}_${num}.fml"
    $fitfluxes_file = "/output/fitfluxes_${file_no_ext}_${num}.fml"

    $h5_file_name = Split-Path $h5_file -leaf
    $fwdsim_file_name = Split-Path $fwdsim_file -leaf
    $fitfluxes_file_name = Split-Path $fitfluxes_file -leaf

    docker exec $image bash -c "USER=nzamboni fwdsim -i $signed_file -o $fwdsim_file -s -H $h5_file -c config_$num"
    docker exec $image bash -c "USER=nzamboni fitfluxes -i $signed_file -o $fitfluxes_file -c config_$num"
    docker cp "${image}:$h5_file" $h5_file_name
    docker cp "${image}:$fwdsim_file" $fwdsim_file_name
    docker cp "${image}:$fitfluxes_file" $fitfluxes_file_name
}

# docker cp FLUX:"/output/signed_ding.fml" "C:\\python_projects\\pysumo\\C13FLUX2\\signed_ding.fml"

# clean the container
# docker exec $image bash -c "rm /output/*"
# docker exec $image bash -c "rm /signed/*"
# docker exec $image bash -c "rm /input/*"
#
# # stop the container
# docker stop $image
#
# # combine hdf5 jacobians via a python script
# python combine_hdf.py $file_no_ext

