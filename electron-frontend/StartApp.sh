. ../.env

docker pull pavtiger/converter:latest
docker pull pavtiger/segmentation:latest

echo $WORKDIR
rm -rf $WORKDIR
mkdir $WORKDIR
./node_modules/electron/dist/electron ./main & docker compose up
