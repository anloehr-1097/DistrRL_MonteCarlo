return {
    {
        type = 'python',
        request = 'launch',
        name = 'Debug Module',
        -- program = "hello_world.py",
        module = 'src',
        pythonPath = function()
            return os.getenv('VIRTUAL_ENV') .. "/bin/python"-- Specify your Python interpreter path
        end,
    },

    {
        type = 'python',
        request = 'launch',
        name = 'Debug Tests',
        module = 'unittest',
        pythonPath = function()
            return os.getenv('VIRTUAL_ENV') .. "/bin/python"-- Specify your Python interpreter path
        end,
    }
}
