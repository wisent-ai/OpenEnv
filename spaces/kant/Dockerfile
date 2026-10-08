FROM rust:slim AS build

# The revision the server is built from. A Space passes its variables to the
# build as arguments, so the Space's settings state it; nothing is built from
# a branch whose contents can change under the same name. An unset variable
# fails this step, which names it.
ARG KANT_REVISION
RUN test -n "$KANT_REVISION"
RUN apt-get update && apt-get install --yes --no-install-recommends git ca-certificates pkg-config \
 && cargo install --locked --git https://github.com/wisent-ai/OpenEnv --rev "$KANT_REVISION" kantbench

FROM debian:stable-slim
COPY --from=build /usr/local/cargo/bin/kant /usr/local/bin/kant
WORKDIR /app

# The settings document every payoff, round count and session limit comes
# from: the Space variable KANT_SETTINGS holds it whole, so the numbers a
# public run plays are the ones its operator stated there.
ARG KANT_SETTINGS
RUN test -n "$KANT_SETTINGS" && printf '%s' "$KANT_SETTINGS" > settings.json

# The address the server listens on is the Space's: Hugging Face routes a
# Docker Space to the port its README's app_port names, or to the default it
# states (https://huggingface.co/docs/hub/spaces-sdks-docker), so the Space
# variable KANT_LISTEN states host and port to match. Unset, the build stops
# here.
ARG KANT_LISTEN
RUN test -n "$KANT_LISTEN"
ENV KANT_LISTEN=$KANT_LISTEN

CMD kant serve --settings settings.json --listen "$KANT_LISTEN"
