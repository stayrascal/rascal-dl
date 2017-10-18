import unittest
from generator import randomdata


class DataGeneratorTest(unittest.TestCase):
    def test_random_text(self):
        print(randomdata.random_text())
        print(randomdata.random_text())
        print(randomdata.random_text())

    def test_random_font(self):
        for i in range(10):
            print(randomdata.random_font())

if __name__ == '__main__':
    unittest.main()
