FROM postgres:13-bullseye

COPY build.sh initdb.sh requirements.sh /tmp/
RUN bash /tmp/build.sh

HEALTHCHECK --interval=1s --timeout=1s --start-period=1s --retries=30 CMD psql \
    --single-transaction \
    --user=$POSTGRES_USER \
    --dbname=$POSTGRES_DB \
    --host=localhost \
    --command="SELECT version();"
