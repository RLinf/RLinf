# Scripts

这些脚本是把之前排查和联调用过的 `curl` 请求整理出来，方便后面直接测试平台接口。

## 前置条件

至少设置：

```bash
export INFINI_API_KEY="sk-xxxxxxxx"
```

可选：

```bash
export INFINI_BASE_URL="https://cloud.infini-ai.com"
```

## 查询类脚本

查看框架：

```bash
./scripts/curl_framework_list.sh
```

查看自定义镜像：

```bash
./scripts/curl_image_build_list.sh
```

查看存储卷：

```bash
./scripts/curl_volume_list.sh
```

查看资源池：

```bash
./scripts/curl_pool_list.sh
```

查看某个区域的资源详情：

```bash
REGION_ID="rg-da5azznpr4i4fdrx" ./scripts/curl_pool_get_resource_detail.sh
```

查看某个资源池下的 load spec：

```bash
POOL_ID="po-da73jexmoe4rfgej" RESOURCE_TYPE=1 ./scripts/curl_load_spec_list.sh
```

查看任务列表：

```bash
./scripts/curl_job_list.sh
```

查看闲时训练服务可用区：

```bash
./scripts/curl_train_service_idle_resources_region_list.sh
```

## 创建任务脚本

创建浙江D Spot 普通训练任务：

```bash
./scripts/curl_job_create_zhejiangd_spot.sh
```

只生成闲时训练服务方案：

```bash
./scripts/curl_train_plan_idle_resources_generate.sh
```

用已有 `train_plan_id` 创建闲时训练服务任务：

```bash
TRAIN_PLAN_PRE_EXECUTION_ID="..." \
TRAIN_PLAN_ID="..." \
./scripts/curl_train_service_idle_resources_create.sh
```

一条命令打通 `generate + create`：

```bash
./scripts/curl_training_server_test.sh
```

## 当前默认值

这些脚本默认已经带了我们最近测试通过或正在联调的浙江D参数：

- `region_id = rg-da5azznpr4i4fdrx`
- `resource_spec_id = rs-dba3vcq2k4po4o5g`
- `image_id = im-dcm6egnmqdgurn6e`
- `framework_id = fw-c6q6a7sfyhoeb5xi`
- `volume_id = vo-dba4je5b5vun473d`
- `mount_path = /mnt/public/`

大多数参数都可以通过环境变量覆盖，例如：

```bash
JOB_NAME="training-server-test-v2" \
ENTRY_POINT="cd /mnt/public/xusi/RLinf-fork-v0.2 && bash run_yaml.sh" \
./scripts/curl_training_server_test.sh
```
