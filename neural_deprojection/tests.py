class TestClass(object):
    def __init__(self, a):
        self.a = a

    def __add__(self, b):
        """
        This overrides addition on this class.

        Args:
            b: some number

        Returns: self.a + b

        """
        return self.a + b

def test_test_class():
    tc = TestClass(1.)
    assert tc + 5. == 6.