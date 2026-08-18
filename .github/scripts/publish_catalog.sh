#!/usr/bin/env bash
# Publishes the catalog of a single provider collected in the working directory.
# Catalogs are published per provider so that a provider that fails to be collected
# does not hold back the others.
#
# Usage: BUCKET=s3://bucket-name publish_catalog.sh <provider>

set -eu

provider=$1
version="$(date +%Y%m%d)-${GITHUB_RUN_NUMBER}"
prefix="${BUCKET}/v3/${provider}"

zip -j catalog.zip "${provider}.csv"
aws s3 cp catalog.zip "${prefix}/${version}/catalog.zip" --acl public-read
aws s3 cp catalog.zip "${prefix}/latest/catalog.zip" --acl public-read
# Written last so that the version never refers to a catalog that is not uploaded yet
echo "${version}" | aws s3 cp - "${prefix}/version" --acl public-read
