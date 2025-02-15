FROM python:3.9-slim
ARG task_type

ENV TASK_TYPE=$task_type
ENV EXECUTE_IN_DOCKER=1

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /test /output \
    && chown algorithm:algorithm /opt/algorithm /test /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm synthrad_model.pt /opt/algorithm/

RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm .env /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm base_algorithm.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@
