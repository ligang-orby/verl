protos:
	protoc --python_out=./ --proto_path=../../protos \
	--mypy_out=./ \
	--experimental_allow_proto3_optional \
	../../protos/pb/*.proto \
	../../protos/pb/v1alpha1/*.proto \
	../../protos/pb/v1alpha2/*.proto \
	../../protos/pb/orby_internal/*.proto \
	../../protos/google/cloud/documentai/v1/*.proto \
	../../protos/google/*/*.proto \
	../../protos/automation_mining/*.proto \
	../../protos/automation_mining/ontology/*.proto \
	../../protos/grpc/reflection/v1alpha/reflection.proto \
	../../protos/fm/*.proto \
	../../protos/pb/demo/*.proto \
	../../protos/common/*.proto

.PHONY: protos
