from gpuhunt.providers.cloudrift import CloudRiftProvider


def test_instance_types():
    cloudrift_provider = CloudRiftProvider()
    instance_types = cloudrift_provider.get()

    # check some common instance types
    has_rtx49_8c_nr = False
    for instance in instance_types:
        if instance.instance_name.startswith('rtx49-8c-nr'):
            has_rtx49_8c_nr = True
            gpu_count = int(instance.instance_name.split('.')[1])
            assert instance.gpu_count == gpu_count
            assert instance.gpu_memory == 24
            assert instance.cpu == 8 * gpu_count
            assert instance.memory == 50 * gpu_count

    assert has_rtx49_8c_nr
